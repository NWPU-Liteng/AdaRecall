#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import random
from collections import Counter, defaultdict

import numpy as np

from medmmb_common import read_jsonl, write_csv


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / float(len(union)) if union else 0.0


def _min_overlap(a: set[str], b: set[str]) -> float:
    denom = min(len(a), len(b))
    return len(a & b) / float(denom) if denom else 0.0


def _same_cross(rows: list[dict]) -> list[dict]:
    by_stage = defaultdict(list)
    for row in rows:
        by_stage[row["stage"]].append(row)
    output = []
    stages = sorted(by_stage)
    for stage in stages:
        same_scores = []
        cross_scores = []
        stage_rows = by_stage[stage]
        for left, right in itertools.combinations(stage_rows, 2):
            if str(left["persona_id"]) == str(right["persona_id"]) and str(left.get("query_type")) == str(right.get("query_type")):
                same_scores.append(_min_overlap(set(left["final_selected_ids"]), set(right["final_selected_ids"])))
        for left in stage_rows:
            for other_stage in stages:
                if other_stage == stage:
                    continue
                for right in by_stage[other_stage]:
                    if str(left["persona_id"]) == str(right["persona_id"]) and str(left.get("query_type")) == str(right.get("query_type")):
                        cross_scores.append(_min_overlap(set(left["final_selected_ids"]), set(right["final_selected_ids"])))
        output.append(
            {
                "stage": stage,
                "same_stage_overlap": float(np.mean(same_scores)) if same_scores else 0.0,
                "cross_stage_overlap": float(np.mean(cross_scores)) if cross_scores else 0.0,
                "same_minus_cross": (float(np.mean(same_scores)) if same_scores else 0.0) - (float(np.mean(cross_scores)) if cross_scores else 0.0),
                "n_same_pairs": len(same_scores),
                "n_cross_pairs": len(cross_scores),
            }
        )
    return output


def _shuffle_within_strata(rows: list[dict], rng: random.Random) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["persona_id"]), str(row.get("query_type", "unknown")))].append(dict(row))
    output = []
    for items in grouped.values():
        labels = [row["stage"] for row in items]
        rng.shuffle(labels)
        for row, stage in zip(items, labels):
            row["stage"] = stage
            output.append(row)
    return output


def _pair_recurrence(rows: list[dict]) -> list[dict]:
    by_stage = defaultdict(list)
    for row in rows:
        by_stage[row["stage"]].append(row)
    global_train = Counter()
    stage_train = defaultdict(Counter)
    heldout_pairs = defaultdict(list)
    by_persona = defaultdict(list)
    for row in rows:
        by_persona[str(row["persona_id"])].append(row)
    for persona_rows in by_persona.values():
        persona_rows.sort(key=lambda row: (str(row.get("query_session_id")), str(row.get("query_id")), int(row.get("stage_index", 0))))
        split = max(1, int(len(persona_rows) * 0.7))
        train = persona_rows[:split]
        heldout = persona_rows[split:]
        for row in train:
            ids = sorted(set(row["final_selected_ids"]))
            for pair in itertools.combinations(ids, 2):
                global_train[pair] += 1
                stage_train[row["stage"]][pair] += 1
        for row in heldout:
            ids = sorted(set(row["final_selected_ids"]))
            heldout_pairs[row["stage"]].extend(list(itertools.combinations(ids, 2)))
    rows_out = []
    global_total = sum(global_train.values())
    vocab = max(1, len(global_train))
    for stage in sorted(by_stage):
        pairs = heldout_pairs.get(stage, [])
        if not pairs:
            rows_out.append({"stage": stage, "heldout_pairs": 0, "matching_stage_pair_recall": 0.0, "global_pair_recall": 0.0, "matching_stage_log_likelihood": 0.0, "global_log_likelihood": 0.0})
            continue
        stage_counter = stage_train[stage]
        stage_hits = sum(1 for pair in pairs if pair in stage_counter)
        global_hits = sum(1 for pair in pairs if pair in global_train)
        stage_total = sum(stage_counter.values())
        stage_ll = sum(math.log((stage_counter[pair] + 1) / float(stage_total + vocab)) for pair in pairs) / len(pairs)
        global_ll = sum(math.log((global_train[pair] + 1) / float(global_total + vocab)) for pair in pairs) / len(pairs)
        rows_out.append(
            {
                "stage": stage,
                "heldout_pairs": len(pairs),
                "matching_stage_pair_recall": stage_hits / float(len(pairs)),
                "global_pair_recall": global_hits / float(len(pairs)),
                "matching_stage_log_likelihood": stage_ll,
                "global_log_likelihood": global_ll,
            }
        )
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute O1 same/cross and stage-shuffle null models.")
    parser.add_argument("--stage-trace", required=True)
    parser.add_argument("--num-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--output-prefix", default="aggregate/O1", help="Output prefix without metric suffix. Default preserves formal filenames.")
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.stage_trace) if not row.get("error")]
    if not rows:
        raise RuntimeError("No valid O1 stage rows.")
    output_prefix = args.output_prefix.rstrip("_")
    observed = _same_cross(rows)
    write_csv(f"{output_prefix}_same_cross_stage_stats.csv", observed)
    obs_stat = float(np.mean([row["same_minus_cross"] for row in observed])) if observed else 0.0
    rng = random.Random(args.seed)
    null_rows = []
    null_values = []
    for idx in range(args.num_permutations):
        shuffled = _shuffle_within_strata(rows, random.Random(rng.randint(0, 10**9)))
        stat_rows = _same_cross(shuffled)
        stat = float(np.mean([row["same_minus_cross"] for row in stat_rows])) if stat_rows else 0.0
        null_values.append(stat)
        null_rows.append({"permutation": idx, "statistic": stat})
    null_values_sorted = sorted(null_values)
    p_value = (sum(1 for value in null_values if value >= obs_stat) + 1) / float(len(null_values) + 1)
    null_mean = float(np.mean(null_values)) if null_values else 0.0
    null_std = float(np.std(null_values)) if null_values else 0.0
    effect = (obs_stat - null_mean) / null_std if null_std > 0 else 0.0
    for row in null_rows:
        row.update(
            {
                "observed_statistic": obs_stat,
                "null_mean": null_mean,
                "null_ci95_low": null_values_sorted[int(0.025 * len(null_values_sorted))],
                "null_ci95_high": null_values_sorted[int(0.975 * len(null_values_sorted))],
                "empirical_p_value": p_value,
                "standardized_effect_size": effect,
            }
        )
    write_csv(f"{output_prefix}_stage_shuffle_null.csv", null_rows)
    write_csv(f"{output_prefix}_pair_recurrence.csv", _pair_recurrence(rows))
    print(f"o1_null={output_prefix}_stage_shuffle_null.csv")


if __name__ == "__main__":
    main()
