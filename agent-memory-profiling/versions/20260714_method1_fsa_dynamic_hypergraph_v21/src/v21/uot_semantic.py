from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter_ns

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TransportResult:
    plan: torch.Tensor
    iterations: int
    potential_delta: float
    plan_delta: float
    converged: bool
    log_u: torch.Tensor
    log_v: torch.Tensor


def _normalize_rows(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, dim=1, eps=1e-12)


def primary_convergence_flag(potential_delta: float, tolerance: float, iterations: int, max_iterations: int) -> bool:
    """The preregistered primary criterion; secondary diagnostics never change it."""
    return bool(
        isfinite(potential_delta)
        and potential_delta <= tolerance
        and 1 <= iterations <= max_iterations
    )


def farthest_point_prototypes(vectors: np.ndarray, count: int, seed: int, device: str) -> np.ndarray:
    """Deterministic spherical farthest-point initialization without K-means."""
    if count < 2 or count > len(vectors):
        raise ValueError("invalid prototype count")
    x = _normalize_rows(torch.as_tensor(vectors, dtype=torch.float32, device=device))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    chosen = [int(torch.randint(len(x), (1,), generator=generator, device=device).item())]
    best_distance = 1.0 - x @ x[chosen[0]].unsqueeze(1)
    best_distance = best_distance.squeeze(1)
    for _ in range(1, count):
        index = int(torch.argmax(best_distance).item())
        chosen.append(index)
        distance = 1.0 - x @ x[index].unsqueeze(1)
        best_distance = torch.minimum(best_distance, distance.squeeze(1))
    return x[chosen].detach().cpu().numpy()


def robust_cost_scale(vectors: np.ndarray, prototypes: np.ndarray, device: str) -> float:
    x = _normalize_rows(torch.as_tensor(vectors, dtype=torch.float32, device=device))
    p = _normalize_rows(torch.as_tensor(prototypes, dtype=torch.float32, device=device))
    cost = (1.0 - x @ p.T).clamp_min(0.0)
    return float(torch.median(cost).item())


def _plan(log_kernel: torch.Tensor, log_u: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_kernel + log_u[:, None] + log_v[None, :])


def log_unbalanced_sinkhorn(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    epsilon: float,
    tau_memory: float,
    tau_prototype: float,
    max_iterations: int,
    tolerance: float,
) -> TransportResult:
    if epsilon <= 0 or tau_memory <= 0 or tau_prototype <= 0:
        raise ValueError("positive epsilon and tau are required")
    if (a <= 0).any() or (b <= 0).any() or not torch.isfinite(cost).all():
        raise ValueError("finite positive transport inputs are required")
    log_kernel = -cost / epsilon
    log_a, log_b = a.log(), b.log()
    rho_m = tau_memory / (tau_memory + epsilon)
    rho_p = tau_prototype / (tau_prototype + epsilon)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    potential_delta = float("inf")
    plan_delta = float("inf")
    previous_plan: torch.Tensor | None = None
    for iteration in range(1, max_iterations + 1):
        previous_log_u, previous_log_v = log_u, log_v
        log_u = rho_m * (log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1))
        log_v = rho_p * (log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0))
        potential_delta = float(
            torch.maximum((log_u - previous_log_u).abs().max(), (log_v - previous_log_v).abs().max()).item()
        )
        current_plan = _plan(log_kernel, log_u, log_v)
        if previous_plan is not None:
            plan_delta = float((current_plan - previous_plan).abs().max().item())
        if primary_convergence_flag(potential_delta, tolerance, iteration, max_iterations):
            return TransportResult(
                plan=current_plan,
                iterations=iteration,
                potential_delta=potential_delta,
                plan_delta=plan_delta,
                converged=True,
                log_u=log_u,
                log_v=log_v,
            )
        previous_plan = current_plan
    return TransportResult(
        plan=current_plan,
        iterations=max_iterations,
        potential_delta=potential_delta,
        plan_delta=plan_delta,
        converged=False,
        log_u=log_u,
        log_v=log_v,
    )


def _logdet_diversity_step(prototypes: torch.Tensor, step_size: float, steps: int) -> torch.Tensor:
    """Projected ascent of logdet(P P^T + ridge I), a diversity constraint."""
    result = prototypes.detach()
    ridge = 1e-4
    for _ in range(steps):
        candidate = result.detach().requires_grad_(True)
        gram = candidate @ candidate.T + ridge * torch.eye(len(candidate), device=candidate.device, dtype=candidate.dtype)
        sign, logabsdet = torch.linalg.slogdet(gram)
        if float(sign) <= 0 or not torch.isfinite(logabsdet):
            break
        gradient, = torch.autograd.grad(logabsdet, candidate)
        result = _normalize_rows(candidate + step_size * gradient)
    return result.detach()


def normalize_semantic_incidence(plan: torch.Tensor, prototype_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a raw UOT plan into the separately specified H_sem incidence."""
    if not torch.isfinite(plan).all() or (plan < 0).any():
        raise FloatingPointError("cannot normalize a non-finite or negative UOT plan")
    full = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
    semantic = full[:, :prototype_count]
    semantic_mass = semantic.sum(dim=1, keepdim=True)
    if (semantic_mass <= 1e-12).any():
        raise FloatingPointError("UOT semantic mass vanished for at least one memory")
    return semantic / semantic_mass, full


def normalize_semantic_log_weights(log_weights: torch.Tensor, prototype_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable H_sem normalization when raw float32 transport entries underflow."""
    if not torch.isfinite(log_weights).all():
        raise FloatingPointError("cannot normalize non-finite UOT log weights")
    full = torch.softmax(log_weights, dim=1)
    semantic = torch.softmax(log_weights[:, :prototype_count], dim=1)
    return semantic, full


def sparse_memberships(incidence: np.ndarray, cumulative_mass: float, max_memberships: int) -> list[list[int]]:
    result: list[list[int]] = []
    for row in incidence:
        order = np.argsort(row)[::-1]
        selected: list[int] = []
        mass = 0.0
        for index in order:
            selected.append(int(index))
            mass += float(row[index])
            if mass >= cumulative_mass or len(selected) >= max_memberships:
                break
        result.append(selected)
    return result


def _kl_divergence(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    safe_value = value.clamp_min(1e-30)
    safe_reference = reference.clamp_min(1e-30)
    return torch.sum(value * (safe_value.log() - safe_reference.log()) - value + reference)


def _primal_objective(
    plan: torch.Tensor,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
    tau_memory: float,
    tau_prototype: float,
) -> float:
    reference = a[:, None] * b[None, :]
    row_mass = plan.sum(dim=1)
    column_mass = plan.sum(dim=0)
    value = (
        (plan * cost).sum()
        + epsilon * _kl_divergence(plan, reference)
        + tau_memory * _kl_divergence(row_mass, a)
        + tau_prototype * _kl_divergence(column_mass, b)
    )
    return float(value.item())


def _dual_surrogate(
    transport: TransportResult,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
    tau_memory: float,
    tau_prototype: float,
) -> tuple[float, bool]:
    """Evaluate the KL-UOT dual from the log scaling potentials.

    Clipping only protects the diagnostic from exponent overflow and is reported
    explicitly, so a clipped value is a dual surrogate rather than a claimed
    exact dual objective.
    """
    f = epsilon * (transport.log_u - a.log())
    g = epsilon * (transport.log_v - b.log())
    scaled_f = -f / tau_memory
    scaled_g = -g / tau_prototype
    clipped = bool((scaled_f.abs() > 80).any().item() or (scaled_g.abs() > 80).any().item())
    term_memory = -tau_memory * torch.sum(a * (torch.exp(scaled_f.clamp(-80, 80)) - 1.0))
    term_prototype = -tau_prototype * torch.sum(b * (torch.exp(scaled_g.clamp(-80, 80)) - 1.0))
    # exp((f + g - c) / epsilon) integrated against a x b equals plan.sum().
    term_entropy = -epsilon * (transport.plan.sum() - (a.sum() * b.sum()))
    return float((term_memory + term_prototype + term_entropy).item()), clipped


def diagnostics(
    prototypes: np.ndarray,
    incidence: np.ndarray,
    full_incidence: np.ndarray,
    transport: TransportResult,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
    tau_memory: float,
    tau_prototype: float,
    elapsed_ms: float,
) -> dict:
    masses = incidence.sum(axis=0)
    entropy = -(incidence * np.log(np.maximum(incidence, 1e-12))).sum(axis=1)
    cosine = prototypes @ prototypes.T
    off_diagonal = cosine[~np.eye(len(prototypes), dtype=bool)]
    gram = prototypes @ prototypes.T + 1e-4 * np.eye(len(prototypes))
    raw_plan = transport.plan
    finite = torch.isfinite(raw_plan)
    row_mass = raw_plan.sum(dim=1)
    column_mass = raw_plan.sum(dim=0)
    row_l1 = float((row_mass - a).abs().sum().item())
    column_l1 = float((column_mass - b).abs().sum().item())
    total_mass = float(raw_plan.sum().item())
    raw_semantic_zero_rows = int((raw_plan[:, : len(prototypes)].sum(dim=1) <= 1e-30).sum().item())
    raw_plan_zero_rows = int((raw_plan.sum(dim=1) <= 1e-30).sum().item())
    mass_distribution = masses / np.maximum(masses.sum(), 1e-12)
    effective_prototype_count = float(np.exp(-(mass_distribution * np.log(np.maximum(mass_distribution, 1e-12))).sum()))
    domination = float(masses.max() / np.maximum(masses.sum(), 1e-12))
    max_cosine = float(off_diagonal.max())
    collapse_flag = bool(
        max_cosine >= 0.9999
        or effective_prototype_count < max(2.0, 0.1 * len(prototypes))
        or domination >= 0.5
    )
    dual_surrogate, dual_surrogate_clipped = _dual_surrogate(
        transport, cost, a, b, epsilon, tau_memory, tau_prototype
    )
    row_sum_error = np.abs(incidence.sum(axis=1) - 1.0)
    return {
        "prototype_count": int(len(prototypes)),
        "prototype_pairwise_cosine_mean": float(off_diagonal.mean()),
        "prototype_pairwise_cosine_max": max_cosine,
        "prototype_logdet": float(np.linalg.slogdet(gram)[1]),
        "prototype_effective_count": effective_prototype_count,
        "prototype_collapse_flag": collapse_flag,
        "effective_mass_min": float(masses.min()),
        "effective_mass_max": float(masses.max()),
        "effective_mass_mean": float(masses.mean()),
        "assignment_entropy_mean": float(entropy.mean()),
        "effective_memberships_mean": float(np.exp(entropy).mean()),
        "single_prototype_domination": domination,
        "dustbin_mean_mass": float(full_incidence[:, -1].mean()),
        "transport_iterations": int(transport.iterations),
        "transport_potential_delta": float(transport.potential_delta),
        "transport_plan_delta": float(transport.plan_delta),
        "transport_converged": bool(transport.converged),
        "primal_objective": _primal_objective(
            raw_plan, cost, a, b, epsilon, tau_memory, tau_prototype
        ),
        "dual_surrogate": dual_surrogate,
        "dual_surrogate_clipped": dual_surrogate_clipped,
        "row_marginal_l1_deviation": row_l1,
        "column_marginal_l1_deviation": column_l1,
        "marginal_mass_residual": 0.5 * (row_l1 + column_l1),
        "finite_ratio": float(finite.float().mean().item()),
        "minimum_plan_value": float(raw_plan.min().item()),
        "maximum_plan_value": float(raw_plan.max().item()),
        "effective_transported_mass": total_mass,
        "raw_semantic_underflow_rows": raw_semantic_zero_rows,
        "raw_plan_underflow_rows": raw_plan_zero_rows,
        "uot_plan_row_marginals_forced_equal": False,
        "incidence_row_sum_error_max": float(row_sum_error.max()),
        "incidence_row_sum_error_mean": float(row_sum_error.mean()),
        "build_ms": float(elapsed_ms),
    }


def fit_uot(
    vectors: np.ndarray,
    initial_prototypes: np.ndarray,
    *,
    epsilon: float,
    tau_memory: float,
    tau_prototype: float,
    dustbin_prior_mass: float,
    dustbin_cost: float,
    outer_iterations: int,
    sinkhorn_iterations: int,
    sinkhorn_tolerance: float,
    diversity_mode: str,
    diversity_step_size: float,
    diversity_steps: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    started = perf_counter_ns()
    x = _normalize_rows(torch.as_tensor(vectors, dtype=torch.float32, device=device))
    p = _normalize_rows(torch.as_tensor(initial_prototypes, dtype=torch.float32, device=device))
    n, k = x.shape[0], p.shape[0]
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
    b = torch.full((k + 1,), (1.0 - dustbin_prior_mass) / k, dtype=torch.float32, device=device)
    b[-1] = dustbin_prior_mass
    transport: TransportResult | None = None
    for _ in range(outer_iterations):
        cost = (1.0 - x @ p.T).clamp_min(0.0)
        cost = torch.cat([cost, torch.full((n, 1), dustbin_cost, device=device)], dim=1)
        transport = log_unbalanced_sinkhorn(
            cost, a, b, epsilon=epsilon, tau_memory=tau_memory, tau_prototype=tau_prototype,
            max_iterations=sinkhorn_iterations, tolerance=sinkhorn_tolerance,
        )
        if not torch.isfinite(transport.plan).all() or (transport.plan < 0).any():
            raise FloatingPointError("invalid UOT plan")
        semantic_plan = transport.plan[:, :k]
        mass = semantic_plan.sum(dim=0)
        proposed = semantic_plan.T @ x / mass[:, None].clamp_min(1e-12)
        p = _normalize_rows(proposed)
        if diversity_mode == "logdet_proximal":
            p = _logdet_diversity_step(p, diversity_step_size, diversity_steps)
        elif diversity_mode != "none":
            raise ValueError(f"unknown diversity mode: {diversity_mode}")
    final_cost = (1.0 - x @ p.T).clamp_min(0.0)
    final_cost = torch.cat([final_cost, torch.full((n, 1), dustbin_cost, device=device)], dim=1)
    transport = log_unbalanced_sinkhorn(
        final_cost, a, b, epsilon=epsilon, tau_memory=tau_memory, tau_prototype=tau_prototype,
        max_iterations=sinkhorn_iterations, tolerance=sinkhorn_tolerance,
    )
    final_log_weights = -final_cost / epsilon + transport.log_u[:, None] + transport.log_v[None, :]
    incidence, full = normalize_semantic_log_weights(final_log_weights, k)
    prototype_np = p.detach().cpu().numpy()
    incidence_np = incidence.detach().cpu().numpy()
    full_np = full.detach().cpu().numpy()
    return prototype_np, incidence_np, diagnostics(
        prototype_np,
        incidence_np,
        full_np,
        transport,
        final_cost,
        a,
        b,
        epsilon,
        tau_memory,
        tau_prototype,
        (perf_counter_ns() - started) / 1_000_000.0,
    )
