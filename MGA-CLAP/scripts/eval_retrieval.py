#!/usr/bin/env python3
"""Evaluate text-to-audio retrieval runs on Clotho (or similar) metrics."""
from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

# Ensure FAISS-related temp dirs exist for numpy/pandas operations.
_TMPDIR = os.environ.get("TMPDIR")
if not _TMPDIR or not Path(_TMPDIR).exists():
    fallback_tmp = Path(__file__).resolve().parents[1] / ".tmp"
    fallback_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(fallback_tmp)

import json

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("eval_retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Parquet/CSV/JSONL file with retrieval results.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Mapping from query_id to relevant audio_ids.")
    parser.add_argument("--rank-col", type=str, default="rerank_rank", help="Rank column for evaluated run.")
    parser.add_argument("--baseline-rank-col", type=str, default="baseline_rank", help="Rank column for MGA baseline.")
    parser.add_argument("--score-col", type=str, default="combined_score")
    parser.add_argument("--baseline-score-col", type=str, default="score")
    parser.add_argument("--query-col", type=str, default="query_id")
    parser.add_argument("--audio-col", type=str, default="audio_id")
    parser.add_argument("--k", nargs="*", type=int, default=(1, 5, 10), help="Recall@K values to report.")
    parser.add_argument("--map-k", type=int, default=10, help="mAP cutoff.")
    parser.add_argument("--ndcg-k", type=int, default=10, help="nDCG cutoff.")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap samples for 95% CI (0 disables).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    if suffix == ".jsonl":
        records: List[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                query_id = entry.get("query_id")
                query_text = entry.get("query_text")
                for item in entry.get("results", []):
                    row = {
                        "query_id": query_id,
                        "query_text": query_text,
                        "audio_id": item.get("audio_id"),
                        "audio_path": item.get("audio_path"),
                        "score": item.get("baseline_score"),
                        "step_score": item.get("step_score"),
                        "combined_score": item.get("combined_score"),
                        "baseline_rank": item.get("baseline_rank"),
                        "step_rank": item.get("step_rank"),
                        "rerank_rank": item.get("rerank_rank"),
                    }
                    records.append(row)
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported results format: {suffix}")


def load_ground_truth(path: Path, query_col: str, audio_col: str) -> Dict[str, List[str]]:
    suffix = path.suffix.lower()
    mapping: Dict[str, List[str]] = {}
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        rows = df.to_dict(orient="records")
    elif suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        rows = df.to_dict(orient="records")
    else:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "data" in data:
            rows = data["data"]  # type: ignore[index]
        else:
            raise ValueError("Unsupported ground-truth format.")

    for row in rows:
        qid = str(row.get(query_col))
        if qid not in mapping:
            mapping[qid] = []
        audio_value = row.get(audio_col)
        if audio_value is None:
            continue
        if isinstance(audio_value, str) and ";" in audio_value:
            mapping[qid].extend([item.strip() for item in audio_value.split(";") if item.strip()])
        elif isinstance(audio_value, (list, tuple)):
            mapping[qid].extend([str(x) for x in audio_value])
        else:
            mapping[qid].append(str(audio_value))
    # Deduplicate
    for qid, items in mapping.items():
        mapping[qid] = sorted(set(items))
    return mapping


@dataclass
class PerQueryStats:
    best_rank: float
    ap_k: float
    ndcg_k: float
    reciprocal_rank: float


def evaluate_run(
    df: pd.DataFrame,
    rank_col: str,
    query_col: str,
    audio_col: str,
    relevant: Dict[str, List[str]],
    k_values: Sequence[int],
    map_k: int,
    ndcg_k: int,
) -> Dict[str, object]:
    per_query: Dict[str, PerQueryStats] = {}
    recall_hits = {k: [] for k in k_values}

    for qid, group in df.groupby(query_col):
        group_sorted = group.sort_values(rank_col)
        rel_ids = relevant.get(qid, [])
        if not rel_ids:
            LOGGER.warning("No relevant items for query_id=%s in ground truth.", qid)
            continue

        hits = group_sorted[audio_col].astype(str).isin(rel_ids).to_numpy()
        positions = np.arange(1, len(group_sorted) + 1)
        hit_positions = positions[hits]
        if hit_positions.size == 0:
            best_rank = float(len(group_sorted) + 1)
            reciprocal_rank = 0.0
        else:
            best_rank = float(hit_positions.min())
            reciprocal_rank = 1.0 / best_rank

        for k in k_values:
            recall_hits[k].append(float(hit_positions.size > 0 and hit_positions.min() <= k))

        ap = 0.0
        hits_found = 0
        for idx, hit in enumerate(hits[:map_k], start=1):
            if hit:
                hits_found += 1
                ap += hits_found / idx
        ap_denominator = min(len(rel_ids), map_k)
        ap_k = ap / ap_denominator if ap_denominator > 0 else 0.0

        dcg = 0.0
        for idx, hit in enumerate(hits[:ndcg_k], start=1):
            if hit:
                dcg += 1.0 / math.log2(idx + 1)
        ideal_hits = min(len(rel_ids), ndcg_k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        per_query[qid] = PerQueryStats(
            best_rank=best_rank,
            ap_k=ap_k,
            ndcg_k=ndcg,
            reciprocal_rank=reciprocal_rank,
        )

    if not per_query:
        raise RuntimeError("No queries evaluated. Check ground truth alignment.")

    best_ranks = np.array([stats.best_rank for stats in per_query.values()])
    ap_values = np.array([stats.ap_k for stats in per_query.values()])
    ndcg_values = np.array([stats.ndcg_k for stats in per_query.values()])
    rr_values = np.array([stats.reciprocal_rank for stats in per_query.values()])

    metrics = {}
    for k in k_values:
        metrics[f"R@{k}"] = 100.0 * np.mean(recall_hits[k])
    metrics["MedianRank"] = float(np.median(best_ranks))
    metrics["MeanRank"] = float(np.mean(best_ranks))
    metrics["MRR"] = float(np.mean(rr_values))
    metrics[f"mAP@{map_k}"] = float(np.mean(ap_values))
    metrics[f"nDCG@{ndcg_k}"] = float(np.mean(ndcg_values))

    metrics_raw = {
        "best_ranks": best_ranks,
        "ap": ap_values,
        "ndcg": ndcg_values,
        "reciprocal_rank": rr_values,
        "recall_hits": recall_hits,
    }
    return {"metrics": metrics, "raw": metrics_raw, "per_query_count": len(per_query)}


def bootstrap_ci(
    raw_stats: Dict[str, np.ndarray],
    recall_hits: Dict[int, List[float]],
    k_values: Sequence[int],
    map_k: int,
    ndcg_k: int,
    n_samples: int,
    seed: int,
) -> Dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    num_queries = raw_stats["best_ranks"].shape[0]
    recalls = {k: np.array(recall_hits[k]) for k in k_values}
    ap_values = raw_stats["ap"]
    ndcg_values = raw_stats["ndcg"]
    rr_values = raw_stats["reciprocal_rank"]
    best_ranks = raw_stats["best_ranks"]

    ci: Dict[str, tuple[float, float]] = {}

    samples: Dict[str, List[float]] = {f"R@{k}": [] for k in k_values}
    samples.update({"MedianRank": [], "MeanRank": [], "MRR": [], f"mAP@{map_k}": [], f"nDCG@{ndcg_k}": []})

    for _ in range(n_samples):
        idx = rng.integers(0, num_queries, size=num_queries)
        for k in k_values:
            samples[f"R@{k}"].append(100.0 * np.mean(recalls[k][idx]))
        boot_ranks = best_ranks[idx]
        samples["MedianRank"].append(float(np.median(boot_ranks)))
        samples["MeanRank"].append(float(np.mean(boot_ranks)))
        samples[f"mAP@{map_k}"].append(float(np.mean(ap_values[idx])))
        samples[f"nDCG@{ndcg_k}"].append(float(np.mean(ndcg_values[idx])))
        samples["MRR"].append(float(np.mean(rr_values[idx])))

    for metric_name, values in samples.items():
        lower = float(np.percentile(values, 2.5))
        upper = float(np.percentile(values, 97.5))
        ci[metric_name] = (lower, upper)
    return ci


def compare_metrics(baseline: Dict[str, float], reranked: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    comparison: Dict[str, Dict[str, float]] = {}
    for key, base_value in baseline.items():
        new_value = reranked.get(key)
        if new_value is None:
            continue
        diff = new_value - base_value
        rel = (diff / base_value * 100.0) if base_value not in (0.0, None) else None
        comparison[key] = {
            "baseline": base_value,
            "reranked": new_value,
            "absolute": diff,
            "relative_percent": rel,
        }
    return comparison


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df = load_results(args.results)
    required_cols = {args.query_col, args.audio_col, args.rank_col, args.baseline_rank_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Results file missing required columns: {missing}")

    ground_truth = load_ground_truth(args.ground_truth, args.query_col, args.audio_col)
    if not ground_truth:
        raise RuntimeError("Ground-truth mapping is empty.")

    evaluated = evaluate_run(
        df,
        args.rank_col,
        args.query_col,
        args.audio_col,
        ground_truth,
        args.k,
        args.map_k,
        args.ndcg_k,
    )
    baseline = evaluate_run(
        df,
        args.baseline_rank_col,
        args.query_col,
        args.audio_col,
        ground_truth,
        args.k,
        args.map_k,
        args.ndcg_k,
    )

    comparison = compare_metrics(baseline["metrics"], evaluated["metrics"])

    result_payload = {
        "results_path": str(args.results),
        "ground_truth": str(args.ground_truth),
        "k": list(args.k),
        "map_k": args.map_k,
        "ndcg_k": args.ndcg_k,
        "baseline_metrics": baseline["metrics"],
        "reranked_metrics": evaluated["metrics"],
        "comparison": comparison,
    }

    if args.bootstrap > 0:
        baseline_ci = bootstrap_ci(
            baseline["raw"],
            baseline["raw"]["recall_hits"],
            args.k,
            args.map_k,
            args.ndcg_k,
            args.bootstrap,
            args.seed,
        )
        rerank_ci = bootstrap_ci(
            evaluated["raw"],
            evaluated["raw"]["recall_hits"],
            args.k,
            args.map_k,
            args.ndcg_k,
            args.bootstrap,
            args.seed + 1,
        )
        result_payload["baseline_ci"] = baseline_ci
        result_payload["reranked_ci"] = rerank_ci

    LOGGER.info("Evaluation summary:")
    for metric, values in comparison.items():
        base = values["baseline"]
        rerank = values["reranked"]
        diff = values["absolute"]
        rel = values["relative_percent"]
        LOGGER.info(
            "%s | baseline=%.3f reranked=%.3f diff=%.3f rel=%s",
            metric,
            base,
            rerank,
            diff,
            f"{rel:.2f}%" if rel is not None else "n/a",
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result_payload, handle, indent=2)
        LOGGER.info("Wrote metrics JSON to %s", args.output_json)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for metric, values in comparison.items():
            row = {
                "metric": metric,
                "baseline": values["baseline"],
                "reranked": values["reranked"],
                "absolute_diff": values["absolute"],
                "relative_percent": values["relative_percent"],
            }
            if args.bootstrap > 0:
                row["baseline_ci"] = result_payload.get("baseline_ci", {}).get(metric)
                row["reranked_ci"] = result_payload.get("reranked_ci", {}).get(metric)
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        LOGGER.info("Wrote metrics CSV to %s", args.output_csv)


if __name__ == "__main__":
    main()
