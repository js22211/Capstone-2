#!/usr/bin/env python3
"""Run MGA-CLAP text→audio retrieval using a prebuilt FAISS index."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

# Ensure FAISS/torch have a writable temporary directory.
_TMPDIR = os.environ.get("TMPDIR")
if not _TMPDIR or not Path(_TMPDIR).exists():
    fallback_tmp = Path(__file__).resolve().parents[1] / ".tmp"
    fallback_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(fallback_tmp)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional at runtime
    pd = None  # type: ignore

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - faiss required
    raise SystemExit("faiss is required for retrieval. Install faiss-cpu or faiss-gpu.") from exc

from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

# Allow importing MGA-CLAP modules when executing from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from models.ase_model import ASE  # noqa: E402


LOGGER = logging.getLogger("search_topk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("settings/inference_example.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--queries", type=Path, required=True, help="Manifest containing text queries.")
    parser.add_argument("--caption-col", type=str, default="caption")
    parser.add_argument("--id-col", type=str, default="query_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--normalize/--no-normalize", dest="normalize", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--index", type=Path, required=True, help="Path to FAISS index.")
    parser.add_argument("--index-meta", type=Path, default=None, help="Path to index metadata JSON (defaults to index with .json).")
    parser.add_argument("--mapping", type=Path, default=None, help="Optional mapping file produced by build_index.py.")
    parser.add_argument("--embeddings", type=Path, default=None, help="Optional embeddings NPZ (used when mapping absent).")
    parser.add_argument("--nprobe", type=int, default=None, help="Override FAISS nprobe (IVF indices).")
    parser.add_argument("--ef-search", type=int, default=None, help="Override HNSW efSearch.")
    parser.add_argument("--use-gpu", action="store_true", help="Run FAISS search on GPU (requires faiss-gpu).")
    parser.add_argument("--results-parquet", type=Path, default=None)
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument("--run-metadata", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


@dataclass
class QueryRecord:
    query_id: str
    query_text: str


class QueryDataset(Dataset):
    def __init__(self, records: Sequence[QueryRecord]):
        self.records = list(records)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.records)

    def __getitem__(self, index: int):  # type: ignore[override]
        rec = self.records[index]
        return rec.query_text, rec.query_id


def load_queries(manifest: Path, caption_col: str, id_col: str, limit: Optional[int]) -> List[QueryRecord]:
    if not manifest.exists():
        raise FileNotFoundError(f"Query manifest not found: {manifest}")

    suffix = manifest.suffix.lower()
    records: List[QueryRecord] = []

    if suffix in {".csv", ".tsv"}:
        if pd is None:
            raise ImportError("pandas is required to parse CSV/TSV query manifests.")
        df = pd.read_csv(manifest, sep="\t" if suffix == ".tsv" else ",")
        if caption_col not in df.columns:
            raise KeyError(f"Column '{caption_col}' not found in {manifest}")
        if id_col and id_col not in df.columns:
            LOGGER.warning("Column '%s' not found; generating sequential query ids.", id_col)
            id_series = pd.Series([None] * len(df))
        else:
            id_series = df[id_col] if id_col in df.columns else pd.Series([None] * len(df))
        for idx, (caption, qid) in enumerate(zip(df[caption_col], id_series)):
            caption_text = str(caption)
            query_id = str(qid) if qid not in (None, "nan", "NaN") else f"q{idx}"
            records.append(QueryRecord(query_id=query_id, query_text=caption_text))
    elif suffix in {".json", ".jsonl"}:
        lines: List[dict]
        if suffix == ".jsonl":
            lines = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        else:
            loaded = json.loads(manifest.read_text())
            if isinstance(loaded, list):
                lines = loaded
            elif isinstance(loaded, dict) and "data" in loaded:
                lines = loaded["data"]  # type: ignore[index]
            else:
                raise ValueError("Unsupported JSON structure for queries")
        for idx, item in enumerate(lines):
            if caption_col not in item:
                raise KeyError(f"Key '{caption_col}' missing in query record: {item}")
            caption_text = str(item[caption_col])
            query_id_val = item.get(id_col)
            query_id = str(query_id_val) if query_id_val not in (None, "") else f"q{idx}"
            records.append(QueryRecord(query_id=query_id, query_text=caption_text))
    elif suffix == ".txt":
        for idx, line in enumerate(manifest.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            records.append(QueryRecord(query_id=f"q{idx}", query_text=line))
    else:
        raise ValueError(f"Unsupported query file format: {suffix}")

    if limit is not None:
        records = records[:limit]

    LOGGER.info("Loaded %d queries", len(records))
    return records


def prepare_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> ASE:
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    model = ASE(config)
    # Ensure compatibility with checkpoints requiring full unpickling
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered_state.keys())
    unexpected = set(state_dict.keys()) - model_keys
    if unexpected:
        LOGGER.warning("Ignoring %d unexpected keys when loading checkpoint (e.g., %s)", len(unexpected), next(iter(unexpected)))
    if missing:
        LOGGER.warning("%d keys missing from checkpoint (e.g., %s) — continuing", len(missing), next(iter(missing)))
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    return model


def embed_queries(model: ASE, dataset: QueryDataset, batch_size: int, num_workers: int, device: torch.device, normalize: bool) -> tuple[np.ndarray, List[str], List[str]]:
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    vectors: List[np.ndarray] = []
    query_ids: List[str] = []
    query_texts: List[str] = []

    with torch.inference_mode():
        for texts, ids in tqdm(dataloader, desc="Encoding queries", unit="batch"):
            word_embeds, attn_mask = None, None
            _, word_embeds, attn_mask = model.encode_text(list(texts))
            query_embeds = model.msc(word_embeds, model.codebook, attn_mask)
            if normalize:
                query_embeds = F.normalize(query_embeds, dim=-1)
            query_embeds = query_embeds.to(torch.float32).cpu().numpy()
            vectors.append(query_embeds)
            query_ids.extend(ids)
            query_texts.extend(texts)

    if vectors:
        matrix = np.concatenate(vectors, axis=0)
    else:
        matrix = np.empty((0, model.codebook.shape[1]), dtype=np.float32)

    return matrix, query_ids, query_texts


def load_mapping(
    mapping_path: Optional[Path],
    embeddings_path: Optional[Path],
    embedding_key: str,
) -> tuple[np.ndarray, List[str], List[str]]:
    audio_ids: List[str]
    audio_paths: List[str]
    index_ids: np.ndarray

    if mapping_path and mapping_path.exists():
        suffix = mapping_path.suffix.lower()
        if suffix == ".parquet":
            if pd is None:
                raise ImportError("pandas is required to read Parquet mapping files.")
            df = pd.read_parquet(mapping_path)
            if "index_id" not in df.columns:
                raise KeyError("Mapping file must contain 'index_id' column.")
            index_ids = df["index_id"].to_numpy(dtype=np.int64)
            audio_ids = [str(x) for x in df["audio_id"].tolist()]
            audio_paths = [str(x) for x in df.get("audio_path", [""] * len(df)).tolist()]
        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(mapping_path, sep=delimiter) if pd is not None else None
            if df is None:
                raise ImportError("pandas is required to parse CSV/TSV mapping files.")
            if "index_id" not in df.columns:
                raise KeyError("Mapping file must contain 'index_id' column.")
            index_ids = df["index_id"].to_numpy(dtype=np.int64)
            audio_ids = [str(x) for x in df["audio_id"].tolist()]
            audio_paths = [str(x) for x in df.get("audio_path", [""] * len(df)).tolist()]
        else:
            raise ValueError("Unsupported mapping format. Use Parquet/CSV/TSV.")
        return index_ids, audio_ids, audio_paths

    if embeddings_path is None or not embeddings_path.exists():
        raise FileNotFoundError("Mapping not provided and embeddings file not available.")

    data = np.load(embeddings_path, allow_pickle=True)
    audio_ids_arr = data.get("audio_ids")
    audio_paths_arr = data.get("audio_paths")
    if audio_ids_arr is None:
        raise KeyError("audio_ids missing from embeddings file; supply --mapping instead.")
    audio_ids = [str(x) for x in audio_ids_arr.tolist()]
    audio_paths = [str(x) for x in audio_paths_arr.tolist()] if audio_paths_arr is not None else [""] * len(audio_ids)
    index_ids = np.arange(len(audio_ids), dtype=np.int64)
    return index_ids, audio_ids, audio_paths


def ensure_paths(output_dir: Path, parquet_path: Optional[Path], jsonl_path: Optional[Path], metadata_path: Optional[Path]) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_parquet = parquet_path or (output_dir / "mga_baseline.parquet")
    results_jsonl = jsonl_path or (output_dir / "mga_baseline.jsonl")
    run_metadata = metadata_path or (output_dir / "run_metadata.json")
    return results_parquet, results_jsonl, run_metadata


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        with args.config.open("r") as handle:
            config_tmp = yaml.safe_load(handle)
        ckpt_cfg = config_tmp.get("eval", {}).get("ckpt")
        if not ckpt_cfg:
            raise ValueError("Checkpoint required via --checkpoint or config eval.ckpt")
        checkpoint_path = Path(ckpt_cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    index_meta_path = args.index_meta or args.index.with_suffix(".json")
    if not index_meta_path.exists():
        LOGGER.warning("Index metadata file %s not found; continuing without it.", index_meta_path)
        index_meta = {}
    else:
        index_meta = json.loads(index_meta_path.read_text())

    embeddings_path = args.embeddings or Path(index_meta.get("embeddings", "")) if index_meta else args.embeddings
    if embeddings_path and not Path(embeddings_path).exists():
        LOGGER.warning("Embeddings file %s not found; will rely solely on mapping.", embeddings_path)
        embeddings_path = None

    mapping_path = args.mapping or (Path(index_meta["mapping"]) if index_meta.get("mapping") else None)

    records = load_queries(args.queries, args.caption_col, args.id_col, args.limit)
    dataset = QueryDataset(records)

    device_str = args.device or (index_meta.get("device") if isinstance(index_meta.get("device"), str) else None)
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model = prepare_model(args.config, checkpoint_path, device)
    query_matrix, query_ids, query_texts = embed_queries(model, dataset, args.batch_size, args.num_workers, device, args.normalize)

    if query_matrix.shape[0] != len(query_ids):
        raise RuntimeError("Mismatch between encoded queries and ids.")

    index = faiss.read_index(str(args.index))
    if args.use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_all_gpus(index)

    if args.nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = args.nprobe
    if args.ef_search is not None and hasattr(index, "hnsw"):
        index.hnsw.efSearch = args.ef_search

    index_ids, audio_ids, audio_paths = load_mapping(mapping_path, embeddings_path, index_meta.get("embedding_key", "clip_embeddings"))
    id_lookup = {int(idx): (aid, apath) for idx, aid, apath in zip(index_ids, audio_ids, audio_paths)}

    topk = args.topk
    distances, indices = index.search(query_matrix.astype(np.float32), topk)

    # Unify score direction: higher is better. For L2 metrics FAISS returns distances
    # where smaller is better; convert to negative distances to keep semantics consistent.
    # Prefer detecting metric from the FAISS index itself; fallback to metadata.
    try:
        is_l2_from_index = hasattr(index, "metric_type") and (index.metric_type == faiss.METRIC_L2)
    except Exception:
        is_l2_from_index = False
    metric_in_meta = str(index_meta.get("metric", "ip")).lower() if isinstance(index_meta, dict) else "ip"
    is_l2_metric = is_l2_from_index or (metric_in_meta == "l2")

    results: List[dict] = []
    for q_idx, (qid, qtext) in enumerate(zip(query_ids, query_texts)):
        for rank in range(topk):
            idx = int(indices[q_idx, rank])
            if idx == -1:
                continue
            raw_score = float(distances[q_idx, rank])
            score = (-raw_score) if is_l2_metric else raw_score
            aid, apath = id_lookup.get(idx, (None, None))
            results.append(
                {
                    "query_idx": q_idx,
                    "query_id": qid,
                    "query_text": qtext,
                    "rank": rank + 1,
                    "index_id": idx,
                    "audio_id": aid,
                    "audio_path": apath,
                    "score": score,
                }
            )

    results_parquet, results_jsonl, run_metadata = ensure_paths(args.output_dir, args.results_parquet, args.results_jsonl, args.run_metadata)

    if pd is not None:
        df = pd.DataFrame(results)
        df.to_parquet(results_parquet, index=False)
        LOGGER.info("Saved retrieval results to %s", results_parquet)
    else:
        LOGGER.warning("pandas not available; skipping Parquet export")

    with results_jsonl.open("w", encoding="utf-8") as handle:
        current_qid = None
        current_payload: dict = {}
        for row in results:
            if row["query_id"] != current_qid:
                if current_payload:
                    handle.write(json.dumps(current_payload) + "\n")
                current_qid = row["query_id"]
                current_payload = {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "results": [],
                }
            current_payload["results"].append(
                {
                    "rank": row["rank"],
                    "index_id": row["index_id"],
                    "audio_id": row["audio_id"],
                    "audio_path": row["audio_path"],
                    "score": row["score"],
                }
            )
        if current_payload:
            handle.write(json.dumps(current_payload) + "\n")
    LOGGER.info("Saved MGA baseline JSONL to %s", results_jsonl)

    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "index": str(args.index),
        "index_meta": str(index_meta_path) if index_meta_path else None,
        "embeddings": str(embeddings_path) if embeddings_path else None,
        "mapping": str(mapping_path) if mapping_path else None,
        "topk": topk,
        "normalize_queries": args.normalize,
        "device": device_str,
        "num_queries": len(records),
        "faiss_use_gpu": args.use_gpu,
        "nprobe": getattr(index, "nprobe", None) if hasattr(index, "nprobe") else None,
        "efSearch": getattr(index.hnsw, "efSearch", None) if hasattr(index, "hnsw") else None,
        "metric": index_meta.get("metric"),
    }

    with run_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    LOGGER.info("Saved run metadata to %s", run_metadata)


if __name__ == "__main__":
    main()
