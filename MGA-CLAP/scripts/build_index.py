#!/usr/bin/env python3
"""Build vector search indices (IVF-PQ / HNSW) from MGA-CLAP audio embeddings."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

# Ensure FAISS can access a writable tmp directory.
_TMPDIR = os.environ.get("TMPDIR")
if not _TMPDIR or not Path(_TMPDIR).exists():
    fallback_tmp = Path(__file__).resolve().parents[1] / ".tmp"
    fallback_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(fallback_tmp)

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - faiss must be installed
    raise SystemExit("faiss is required to build indices. Install faiss-cpu or faiss-gpu.") from exc


LOGGER = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to NPZ file produced by build_audio_embeds.py (must contain clip_embeddings).",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="clip_embeddings",
        help="Key inside the NPZ file that stores the embedding matrix.",
    )
    parser.add_argument(
        "--index-type",
        choices=("ivfpq", "hnsw", "flat"),
        default="ivfpq",
        help="Type of index to build.",
    )
    parser.add_argument(
        "--metric",
        choices=("ip", "l2"),
        default="ip",
        help="Similarity metric for the index (ip: inner product / cosine, l2: Euclidean).",
    )
    parser.add_argument(
        "--normalize/--no-normalize",
        dest="normalize",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="L2-normalize embeddings before indexing (recommended for cosine/IP).",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=1024,
        help="Number of IVF clusters (only for IVFPQ).",
    )
    parser.add_argument(
        "--pq-m",
        type=int,
        default=32,
        help="Number of PQ subvectors (only for IVFPQ).",
    )
    parser.add_argument(
        "--pq-nbits",
        type=int,
        default=8,
        help="Number of bits per PQ code (only for IVFPQ).",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=16,
        help="Number of IVF clusters to probe at search time (only for IVFPQ).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of vectors used for IVF training (defaults to min(100000, N)).",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="Connectivity parameter M for HNSW (only for HNSW).",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=200,
        help="efConstruction for HNSW build (only for HNSW).",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=64,
        help="Default efSearch for HNSW queries (only for HNSW).",
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        required=True,
        help="Path to persist the built FAISS index (e.g., index.ivfpq).",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        required=True,
        help="Path to write the index metadata JSON.",
    )
    parser.add_argument(
        "--output-mapping",
        type=Path,
        default=None,
        help="Optional Parquet/CSV file storing id→(audio_id, audio_path) mapping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for IVF training sampling.",
    )

    return parser.parse_args()


def read_embeddings(embeddings_path: Path, key: str) -> tuple[np.ndarray, List[str], List[str]]:
    data = np.load(embeddings_path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Embedding key '{key}' not found in {embeddings_path}.")
    vectors = np.asarray(data[key])
    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix (num_items, dim).")

    audio_ids = data.get("audio_ids")
    audio_paths = data.get("audio_paths")

    if audio_ids is None:
        audio_ids_list = [str(i) for i in range(vectors.shape[0])]
    else:
        audio_ids_list = [str(x) for x in audio_ids.tolist()]

    if audio_paths is None:
        audio_paths_list = [""] * vectors.shape[0]
    else:
        audio_paths_list = [str(x) for x in audio_paths.tolist()]

    return vectors, audio_ids_list, audio_paths_list


def build_ivfpq(
    xb: np.ndarray,
    ids: np.ndarray,
    metric: str,
    nlist: int,
    m: int,
    nbits: int,
    nprobe: int,
    train_size: Optional[int],
    seed: int,
) -> faiss.IndexIVFPQ:
    d = xb.shape[1]
    quantizer = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
    # Ensure IVFPQ uses the intended metric (IP or L2). The 6-arg ctor is available in FAISS>=1.6.
    metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    try:
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, metric_type)
    except TypeError:
        # Fallback for older bindings: construct then set metric_type if exposed.
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        if hasattr(index, "metric_type"):
            index.metric_type = metric_type

    rng = np.random.default_rng(seed)
    num_train = min(train_size or min(100000, xb.shape[0]), xb.shape[0])
    train_idx = rng.choice(xb.shape[0], size=num_train, replace=False)
    train_vectors = xb[train_idx]

    LOGGER.info("Training IVF-PQ with %d vectors (nlist=%d, m=%d, nbits=%d)", num_train, nlist, m, nbits)
    index.train(train_vectors)
    index.add_with_ids(xb, ids)
    index.nprobe = nprobe
    return index


def build_hnsw(
    xb: np.ndarray,
    ids: np.ndarray,
    metric: str,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> faiss.IndexHNSWFlat:
    d = xb.shape[1]
    metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(d, m, metric_type)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add_with_ids(xb, ids)
    return index


def build_flat(xb: np.ndarray, ids: np.ndarray, metric: str) -> faiss.Index:
    d = xb.shape[1]
    if metric == "ip":
        base = faiss.IndexFlatIP(d)
    else:
        base = faiss.IndexFlatL2(d)
    index = faiss.IndexIDMap(base)
    index.add_with_ids(xb, ids)
    return index


def write_mapping(mapping_path: Path, ids: np.ndarray, audio_ids: List[str], audio_paths: List[str]) -> None:
    if mapping_path.suffix.lower() == ".parquet":
        if pd is None:
            raise ImportError("pandas is required to export the id mapping to Parquet.")
        df = pd.DataFrame({
            "index_id": ids.astype(np.int64),
            "audio_id": audio_ids,
            "audio_path": audio_paths,
        })
        df.to_parquet(mapping_path, index=False)
    elif mapping_path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if mapping_path.suffix.lower() == ".tsv" else ","
        header = delimiter.join(["index_id", "audio_id", "audio_path"]) + "\n"
        with mapping_path.open("w", encoding="utf-8") as handle:
            handle.write(header)
            for idx, aid, apath in zip(ids, audio_ids, audio_paths):
                handle.write(f"{idx}{delimiter}{aid}{delimiter}{apath}\n")
    else:
        raise ValueError("Unsupported mapping extension. Use .parquet, .csv or .tsv.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    vectors, audio_ids, audio_paths = read_embeddings(args.embeddings, args.embedding_key)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    if args.normalize:
        faiss.normalize_L2(vectors)

    if args.index_type == "ivfpq" and args.nlist > vectors.shape[0]:
        raise ValueError("nlist cannot exceed number of vectors.")

    ids = np.arange(vectors.shape[0], dtype=np.int64)

    if args.index_type == "ivfpq":
        index = build_ivfpq(vectors, ids, args.metric, args.nlist, args.pq_m, args.pq_nbits, args.nprobe, args.train_size, args.seed)
    elif args.index_type == "hnsw":
        index = build_hnsw(vectors, ids, args.metric, args.hnsw_m, args.hnsw_ef_construction, args.hnsw_ef_search)
    else:
        index = build_flat(vectors, ids, args.metric)

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output_index))
    LOGGER.info("Wrote index to %s", args.output_index)

    meta = {
        "embeddings": str(args.embeddings),
        "embedding_key": args.embedding_key,
        "index_type": args.index_type,
        "metric": args.metric,
        "normalize": args.normalize,
        "num_items": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]) if vectors.size else None,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
    }
    if args.index_type == "ivfpq":
        meta.update({
            "nlist": args.nlist,
            "pq_m": args.pq_m,
            "pq_nbits": args.pq_nbits,
            "nprobe": args.nprobe,
            "train_size": args.train_size,
            "seed": args.seed,
        })
    elif args.index_type == "hnsw":
        meta.update({
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "hnsw_ef_search": args.hnsw_ef_search,
        })

    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    with args.output_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    LOGGER.info("Wrote metadata to %s", args.output_meta)

    if args.output_mapping:
        args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
        write_mapping(args.output_mapping, ids, audio_ids, audio_paths)
        LOGGER.info("Wrote id mapping to %s", args.output_mapping)
        meta["mapping"] = str(args.output_mapping)


if __name__ == "__main__":
    main()
