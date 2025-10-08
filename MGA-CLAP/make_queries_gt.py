#!/usr/bin/env python3
"""Generate query JSONL and ground-truth JSON from Clotho captions CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-csv", type=Path, required=True, help="Path to Clotho captions CSV.")
    parser.add_argument("--output-queries", type=Path, default=Path("artifacts/val/queries.jsonl"))
    parser.add_argument("--output-gt", type=Path, default=Path("artifacts/val/gt.json"))
    parser.add_argument("--id-col", type=str, default="file_name", help="CSV column with audio filename.")
    parser.add_argument("--caption-prefix", type=str, default="caption", help="Prefix of caption columns.")
    return parser.parse_args()

def stem_without_ext(filename: str) -> str:
    return Path(filename).stem

def main() -> None:
    args = parse_args()
    if not args.captions_csv.exists():
        raise FileNotFoundError(f"Captions CSV not found: {args.captions_csv}")

    args.output_queries.parent.mkdir(parents=True, exist_ok=True)
    args.output_gt.parent.mkdir(parents=True, exist_ok=True)

    queries: List[dict] = []
    gts: List[dict] = []

    with args.captions_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        caption_cols = [c for c in (reader.fieldnames or []) if c.lower().startswith(args.caption_prefix.lower())]
        if not caption_cols:
            raise ValueError(f"No caption columns starting with '{args.caption_prefix}' in {args.captions_csv}")

        for row in reader:
            fname = row.get(args.id_col)
            if not fname:
                continue
            stem = stem_without_ext(str(fname))
            for i, col in enumerate(caption_cols, start=1):
                cap = row.get(col)
                if cap is None:
                    continue
                qid = f"{stem}#{i}"
                queries.append({"query_id": qid, "caption": str(cap)})
                gts.append({"query_id": qid, "audio_id": stem})

    with args.output_queries.open("w", encoding="utf-8") as out_q:
        for q in queries:
            out_q.write(json.dumps(q, ensure_ascii=False) + "\n")

    with args.output_gt.open("w", encoding="utf-8") as out_g:
        json.dump({"data": gts}, out_g, ensure_ascii=False, indent=2)

    print(f"queries: {len(queries)} | gt: {len(gts)}")
    print(f"queries.jsonl -> {args.output_queries}")
    print(f"gt.json      -> {args.output_gt}")

if __name__ == "__main__":
    main()
