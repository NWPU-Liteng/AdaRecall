#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def norm(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def session_le(left: Any, right: Any) -> bool:
    try:
        return int(float(left)) <= int(float(right))
    except Exception:
        return str(left) <= str(right)


def keypoints(query: dict[str, Any]) -> list[dict[str, Any]]:
    value = query.get("source_key_points") or []
    return value if isinstance(value, list) else []


def best_alignment(kp: dict[str, Any], memories: list[dict[str, Any]], threshold: float) -> tuple[dict[str, Any] | None, str, float]:
    text = str(kp.get("content", ""))
    ntext = norm(text)
    if not ntext:
        return None, "empty_keypoint", 0.0
    candidates = []
    for mem in memories:
        chunks = [str(mem.get("text", ""))]
        for item in mem.get("knowledge_points") or []:
            if isinstance(item, dict):
                chunks.append(str(item.get("content", "")))
                chunks.append(str(item.get("name", "")))
        joined = " ".join(chunks)
        njoined = norm(joined)
        if ntext and ntext in njoined:
            return mem, "exact_containment", 1.0
        score = difflib.SequenceMatcher(None, ntext, njoined).ratio()
        candidates.append((score, mem))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if candidates and candidates[0][0] >= threshold:
        return candidates[0][1], "near_exact_lexical", float(candidates[0][0])
    return (candidates[0][1], "below_threshold_best_candidate", float(candidates[0][0])) if candidates else (None, "no_candidate", 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict source-keypoint aligned subset and M2 budget summary.")
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    queries = {str(q["query_id"]): q for q in read_jsonl(ROOT / "aggregate" / "queries_medmemorybench_en.jsonl")}
    memories = read_jsonl(ROOT / "aggregate" / "memory_items_turn_pair.jsonl")
    by_persona_session = defaultdict(list)
    by_id = {}
    for mem in memories:
        by_id[str(mem["memory_id"])] = mem
        by_persona_session[(str(mem.get("persona_id")), str(mem.get("session_id")))].append(mem)
    m2_rows = read_jsonl(ROOT / "aggregate" / "M2_frontier_budget_invocations.jsonl")
    formal_qids = sorted({str(r["query_id"]) for r in m2_rows})
    alignment_rows = []
    subset_qids = []
    summary_rows = []
    for qid in formal_qids:
        q = queries[qid]
        kps = keypoints(q)
        ok = bool(kps)
        aligned_count = 0
        high_count = 0
        for idx, kp in enumerate(kps):
            source_session = kp.get("session_id")
            candidates = [
                mem for mem in by_persona_session.get((str(q.get("persona_id")), str(source_session)), [])
                if session_le(mem.get("session_id"), q.get("query_session_id"))
            ]
            mem, rule, score = best_alignment(kp, candidates, args.threshold)
            high = bool(mem and rule in {"exact_containment", "near_exact_lexical"} and score >= (1.0 if rule == "exact_containment" else args.threshold))
            if mem:
                aligned_count += 1
            if high:
                high_count += 1
            ok = ok and high
            alignment_rows.append(
                {
                    "query_id": qid,
                    "keypoint_id": idx,
                    "source_keypoint_text": kp.get("content", ""),
                    "memory_id": mem.get("memory_id") if mem else "",
                    "alignment_rule": rule,
                    "alignment_score": score,
                    "source_session_id": source_session,
                    "causally_eligible": bool(mem and session_le(mem.get("session_id"), q.get("query_session_id"))),
                    "high_confidence": high,
                }
            )
        if ok:
            subset_qids.append(qid)
        summary_rows.append(
            {
                "query_id": qid,
                "persona_id": q.get("persona_id"),
                "query_type": q.get("query_type"),
                "n_source_keypoints": len(kps),
                "n_aligned": aligned_count,
                "n_high_confidence": high_count,
                "in_strict_subset": ok,
            }
        )
    write_jsonl(ROOT / "aggregate" / "C_keypoint_alignment_rows.jsonl", alignment_rows)
    write_csv(ROOT / "aggregate" / "C_keypoint_alignment_summary.csv", summary_rows)

    strict_gold = defaultdict(set)
    for row in alignment_rows:
        if row["high_confidence"]:
            strict_gold[row["query_id"]].add(row["memory_id"])
    budget_rows = []
    for budget in sorted({int(r["frontier_budget"]) for r in m2_rows}):
        sub = [r for r in m2_rows if int(r["frontier_budget"]) == budget and str(r["query_id"]) in subset_qids]
        if not sub:
            continue
        recalls = []
        completions = []
        for row in sub:
            gold = strict_gold[str(row["query_id"])]
            selected = set(str(x) for x in row.get("final_selected_ids", []))
            hit = len(gold & selected)
            recalls.append(hit / float(len(gold)) if gold else 0.0)
            completions.append(1.0 if gold and gold <= selected else 0.0)
        budget_rows.append(
            {
                "frontier_budget": budget,
                "n_queries": len(sub),
                "strict_subset_queries": len(subset_qids),
                "strict_subset_coverage": len(subset_qids) / float(len(formal_qids)) if formal_qids else 0.0,
                "StrictKeypointRecall_at_5_mean": sum(recalls) / len(recalls),
                "StrictKeypointCompletion_at_5_mean": sum(completions) / len(completions),
                "RerankerOracleOverlap_at_5_mean": sum(float(r.get("reranker_oracle_overlap_at_5", 0.0)) for r in sub) / len(sub),
                "selection_latency_ms_mean": sum(float(r.get("t_total_selection_ms", 0.0)) for r in sub) / len(sub),
                "rerank_latency_ms_mean": sum(float(r.get("t_rerank_ms", 0.0)) for r in sub) / len(sub),
                "frontier_fill_ratio_mean": sum(float(r.get("frontier_fill_ratio", 0.0)) for r in sub) / len(sub),
            }
        )
    write_csv(ROOT / "aggregate" / "C_strict_evidence_budget_summary.csv", budget_rows)

    packet = []
    for row in alignment_rows:
        if len(packet) >= 120:
            break
        q = queries[row["query_id"]]
        mem = by_id.get(str(row["memory_id"]), {})
        packet.append(
            {
                "query_id": row["query_id"],
                "query_text": q.get("query_text", ""),
                "query_type": q.get("query_type", ""),
                "source_key_point": row["source_keypoint_text"],
                "candidate_aligned_memory_text": str(mem.get("text", ""))[:1200],
                "memory_id": row["memory_id"],
                "source_session": row["source_session_id"],
                "alignment_rule": row["alignment_rule"],
                "alignment_score": row["alignment_score"],
                "human_approve_reject": "",
                "corrected_memory_id": "",
            }
        )
    write_csv(ROOT / "artifacts" / "evidence_review_packet.csv", packet)

    coverage = len(subset_qids) / float(len(formal_qids)) if formal_qids else 0.0
    (ROOT / "reports" / "C_KEYPOINT_ALIGNMENT_AUDIT.md").write_text(
        "\n".join(
            [
                "# C Keypoint Alignment Audit",
                "",
                f"Formal M2 queries considered: {len(formal_qids)}",
                f"Strict Source-Keypoint-Aligned Subset queries: {len(subset_qids)}",
                f"Strict subset coverage: {coverage:.4f}",
                f"Alignment threshold: exact containment OR lexical similarity >= {args.threshold}",
                "",
                "This is deterministic automatic alignment only. It is not human-verified evidence ground truth.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    interp = ["# C Strict Evidence Interpretation", ""]
    if coverage < 0.30:
        interp.append("Strict-subset coverage is low; this result should be treated as appendix material.")
    if budget_rows:
        first = budget_rows[0]["StrictKeypointRecall_at_5_mean"]
        last = budget_rows[-1]["StrictKeypointRecall_at_5_mean"]
        if abs(last - first) < 0.02:
            interp.append("Strict recall is approximately flat across budgets, indicating diminishing evidence returns under this deterministic subset.")
        else:
            interp.append(f"Strict recall changes from {first:.3f} to {last:.3f}; report this as evidence retrieval behavior, not answer-quality improvement.")
    interp.append("Do not use the old session-level Evidence Completion@5 as a main-paper strict evidence figure.")
    (ROOT / "reports" / "C_STRICT_EVIDENCE_INTERPRETATION.md").write_text("\n".join(interp) + "\n", encoding="utf-8")
    (ROOT / "reports" / "EVIDENCE_REVIEW_GUIDE.md").write_text(
        "# Evidence Review Guide\n\nThe packet `artifacts/evidence_review_packet.csv` contains deterministic automatic keypoint alignments for future human review. Leave `human_approve_reject` and `corrected_memory_id` blank until a human reviewer fills them.\n",
        encoding="utf-8",
    )
    print(f"strict_subset_queries={len(subset_qids)} coverage={coverage:.4f}")


if __name__ == "__main__":
    main()
