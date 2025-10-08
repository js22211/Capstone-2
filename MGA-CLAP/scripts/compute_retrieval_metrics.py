#!/usr/bin/env python3
"""Compute Recall@K and MRR for text→audio retrieval results.

Supports two input formats:
- JSONL with one object per query: {"query_id", "results": [{"audio_id", "rank" or "rerank_rank"}, ...]}
- Parquet/CSV table with columns: query_id, audio_id, and either rank or rerank_rank

Writes a single-row or multi-row CSV with metrics per input file.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True, help="One or more result files (JSONL/Parquet/CSV).")
    p.add_argument("--label", type=str, nargs="*", default=None, help="Optional labels aligned to --input (defaults to basename).")
    p.add_argument("--rank-col", type=str, nargs="*", default=None, help="Optional rank column per input (defaults to auto: rerank_rank or rank).")
    p.add_argument("--output-csv", type=Path, required=True, help="Path to write metrics CSV.")
    p.add_argument("--k", type=int, nargs="*", default=(1, 5, 10), help="Recall@K values to compute.")
    return p.parse_args()


def ground_truth_audio_id(query_id: str) -> str:
    return query_id.split("#", 1)[0].strip()


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_metrics_jsonl(records: List[dict], rank_key: Optional[str], ks: Iterable[int]) -> Dict[str, float]:
    ks = tuple(sorted(set(int(k) for k in ks)))
    hits = {k: 0 for k in ks}
    mrr = 0.0
    total = len(records)

    for rec in records:
        target = ground_truth_audio_id(rec.get("query_id", ""))
        results = rec.get("results", [])
        # Auto-detect rank key per record if not specified
        rk = rank_key
        if rk is None and results:
            if "rerank_rank" in results[0]:
                rk = "rerank_rank"
            elif "rank" in results[0]:
                rk = "rank"
            else:
                raise KeyError("Neither 'rank' nor 'rerank_rank' found in JSONL results.")

        best_rank: Optional[int] = None
        for item in results:
            if str(item.get("audio_id", "")).strip() == target:
                val = item.get(rk) if rk else None
                if val is None:
                    continue
                r = int(val)
                if best_rank is None or r < best_rank:
                    best_rank = r

        if best_rank is not None:
            for k in ks:
                if best_rank <= k:
                    hits[k] += 1
            mrr += 1.0 / best_rank

    out = {f"Recall@{k}": (hits[k] / total if total else 0.0) for k in ks}
    out.update({"MRR": (mrr / total if total else 0.0), "Queries": total})
    return out


def compute_metrics_table(path: Path, rank_col: Optional[str], ks: Iterable[int]) -> Dict[str, float]:
    import pandas as pd  # local import to avoid hard dependency for JSONL mode

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported table format: {suffix}")

    # Auto-detect rank column if not provided
    rk = rank_col
    if rk is None:
        if "rerank_rank" in df.columns:
            rk = "rerank_rank"
        elif "rank" in df.columns:
            rk = "rank"
        else:
            raise KeyError("Neither 'rank' nor 'rerank_rank' found in table columns.")

    ks = tuple(sorted(set(int(k) for k in ks)))
    hits = {k: 0 for k in ks}
    mrr = 0.0
    n_queries = int(df["query_id"].nunique()) if "query_id" in df.columns else 0

    for qid, group in df.groupby("query_id"):
        target = ground_truth_audio_id(str(qid))
        # Find best (minimum) rank where audio_id matches target
        sub = group.loc[group["audio_id"].astype(str).str.strip() == target]
        if not sub.empty:
            r = int(sub[rk].min())
            for k in ks:
                if r <= k:
                    hits[k] += 1
            mrr += 1.0 / r

    out = {f"Recall@{k}": (hits[k] / n_queries if n_queries else 0.0) for k in ks}
    out.update({"MRR": (mrr / n_queries if n_queries else 0.0), "Queries": n_queries})
    return out


def compute_for_path(path: Path, rank_col: Optional[str], ks: Iterable[int]) -> Dict[str, float]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        recs = load_jsonl(path)
        return compute_metrics_jsonl(recs, rank_col, ks)
    elif suf in {".parquet", ".csv", ".tsv"}:
        return compute_metrics_table(path, rank_col, ks)
    else:
        raise ValueError(f"Unsupported input format: {suf}")


def main() -> None:
    args = parse_args()
    inputs: List[Path] = args.input
    labels: List[str] = args.label or []
    rank_cols: List[Optional[str]] = args.rank_col or []

    if labels and len(labels) != len(inputs):
        raise SystemExit("--label count must match --input count (or omit to auto-label).")
    if rank_cols and len(rank_cols) != len(inputs):
        raise SystemExit("--rank-col count must match --input count (or omit to auto-detect).")

    rows: List[Tuple] = []
    ks = tuple(args.k)
    header = ["label", "input_path", "Queries"] + [f"Recall@{k}" for k in ks] + ["MRR", "rank_col"]

    for i, path in enumerate(inputs):
        label = labels[i] if i < len(labels) else path.stem
        rank_col = rank_cols[i] if i < len(rank_cols) else None
        metrics = compute_for_path(path, rank_col, ks)
        detected_rank_col = rank_col
        if detected_rank_col is None:
            # Try to infer by inspecting data quickly
            suf = path.suffix.lower()
            if suf == ".jsonl":
                # Peek first line
                recs = load_jsonl(path)
                if recs and recs[0].get("results"):
                    first = recs[0]["results"][0]
                    detected_rank_col = "rerank_rank" if "rerank_rank" in first else ("rank" if "rank" in first else "?")
            else:
                try:
                    import pandas as pd  # noqa: F401
                    df_head_cols = None
                    # We don't load entire file again to keep it light; mark as auto.
                    detected_rank_col = "auto(rerank_rank|rank)"
                except Exception:
                    detected_rank_col = "auto"

        row = [label, str(path), int(metrics.get("Queries", 0))] + [metrics.get(f"Recall@{k}", 0.0) for k in ks] + [metrics.get("MRR", 0.0), detected_rank_col or "auto"]
        rows.append(tuple(row))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Also echo to stdout for convenience
    print(", ".join(header))
    for r in rows:
        pretty = list(r)
        # format floats
        for j in range(3, 3 + len(ks) + 1):
            if isinstance(pretty[j], float):
                pretty[j] = f"{pretty[j]:.4f}"
        print(", ".join(str(x) for x in pretty))


if __name__ == "__main__":
    main()

