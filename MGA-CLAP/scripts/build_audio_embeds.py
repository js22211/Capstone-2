#!/usr/bin/env python3
"""Batch-generate MGA-CLAP audio embeddings for retrieval indexing.

This script loads a pretrained MGA-CLAP (ASE) model, iterates over a list of
audio files, and writes clip-level embeddings to NPZ/Parquet artifacts together
with a metadata JSON manifest describing the embedding specification.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Ensure an accessible temporary directory before heavyweight imports (torch).
_TMPDIR = os.environ.get("TMPDIR")
if not _TMPDIR or not Path(_TMPDIR).exists():
    fallback_tmp = Path(__file__).resolve().parents[1] / ".tmp"
    fallback_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(fallback_tmp)

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - pandas is optional at runtime
    pd = None  # type: ignore

# Make repository modules importable when launched from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from models.ase_model import ASE  # noqa: E402


LOGGER = logging.getLogger("build_audio_embeds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("settings/inference_example.yaml"),
        help="Path to MGA-CLAP inference YAML configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained model checkpoint. Overrides config.eval.ckpt if provided.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest file (CSV/TSV/JSON/JSONL/TXT) describing audio paths.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Optional directory to recursively collect audio files from when manifest is absent.",
    )
    parser.add_argument(
        "--audio-col",
        type=str,
        default="audio_path",
        help="Column/key name that contains audio paths inside the manifest.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="file_name",
        help="Optional column/key to use as a unique identifier. Defaults to basename when missing.",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        nargs="*",
        default=(".wav", ".flac", ".mp3", ".ogg"),
        help="Audio suffixes to accept when scanning directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store embeddings and metadata.",
    )
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Optional explicit path for the output NPZ file.",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="Optional explicit path for the output Parquet file. Requires pandas+pyarrow/fastparquet.",
    )
    parser.add_argument(
        "--spec-json",
        type=Path,
        default=None,
        help="Optional explicit path for the embedding specification JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for audio encoding.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to run inference on. Defaults to config device or auto-detected CUDA.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Data type used when persisting embeddings.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional override for maximum audio duration in seconds. Defaults to config audio_args.max_length.",
    )
    parser.add_argument(
        "--mono/--no-mono",
        dest="mono",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Convert audio to mono (default: True).",
    )
    parser.add_argument(
        "--normalize/--no-normalize",
        dest="normalize",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply L2 normalization to clip embeddings before saving (default: True).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministically truncate audio instead of random cropping when exceeding max duration.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of audio files processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load everything but skip the forward pass. Useful for debugging manifests.",
    )

    args = parser.parse_args()
    return args


@dataclass
class AudioEntry:
    audio_path: Path
    audio_id: str


class AudioDataset(Dataset):
    """Dataset that loads raw audio waveforms and performs light preprocessing."""

    def __init__(
        self,
        entries: Sequence[AudioEntry],
        target_sr: int,
        max_length: Optional[int],
        mono: bool = True,
        deterministic: bool = False,
    ):
        self.entries = list(entries)
        self.target_sr = target_sr
        self.max_length = max_length
        self.mono = mono
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):  # type: ignore[override]
        entry = self.entries[index]
        waveform, sr = torchaudio.load(str(entry.audio_path))

        if self.mono and waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        if sr != self.target_sr:
            waveform = AF.resample(waveform, sr, self.target_sr)

        if self.max_length and waveform.numel() > self.max_length:
            if self.deterministic:
                start = 0
            else:
                start = 0  # use leading segment for determinism in retrieval use-cases
            waveform = waveform[start : start + self.max_length]

        return {
            "waveform": waveform.float(),
            "length": waveform.numel(),
            "audio_id": entry.audio_id,
            "audio_path": str(entry.audio_path),
        }


def collate_batch(batch: Sequence[dict]) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    waveforms = [item["waveform"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    padded = pad_sequence(waveforms, batch_first=True)
    meta = [
        {
            "audio_id": item["audio_id"],
            "audio_path": item["audio_path"],
        }
        for item in batch
    ]
    return padded, lengths, meta


def load_manifest(
    manifest_path: Optional[Path],
    audio_dir: Optional[Path],
    audio_col: str,
    id_col: Optional[str],
    suffixes: Sequence[str],
    limit: Optional[int] = None,
) -> List[AudioEntry]:
    if manifest_path is None and audio_dir is None:
        raise ValueError("Either --manifest or --audio-dir must be provided.")

    entries: List[AudioEntry] = []

    if manifest_path:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        suffix = manifest_path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            if pd is None:
                raise ImportError("pandas is required to parse CSV/TSV manifests.")
            df = pd.read_csv(manifest_path, sep="\t" if suffix == ".tsv" else ",")
            if audio_col not in df.columns:
                raise KeyError(f"Column '{audio_col}' not found in manifest {manifest_path}.")
            if id_col and id_col not in df.columns:
                raise KeyError(f"Column '{id_col}' not found in manifest {manifest_path}.")
            for _, row in df.iterrows():
                audio_path = Path(row[audio_col])
                audio_id = str(row[id_col]) if id_col and not pd.isna(row[id_col]) else audio_path.stem
                entries.append(AudioEntry(audio_path=audio_path, audio_id=audio_id))
        elif suffix in {".json", ".jsonl"}:
            records: List[dict]
            if suffix == ".jsonl":
                records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
            else:
                data = json.loads(manifest_path.read_text())
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        records = data["data"]
                    else:
                        records = [data]
                elif isinstance(data, list):
                    records = data
                else:
                    raise ValueError(f"Unable to parse JSON manifest structure: {manifest_path}")
            for item in records:
                if audio_col not in item:
                    raise KeyError(f"Key '{audio_col}' missing in manifest record: {item}")
                audio_path = Path(item[audio_col])
                audio_id_value = item.get(id_col) if id_col else None
                audio_id = str(audio_id_value) if audio_id_value not in (None, "") else audio_path.stem
                entries.append(AudioEntry(audio_path=audio_path, audio_id=audio_id))
        elif suffix == ".txt":
            for line in manifest_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                audio_path = Path(line)
                entries.append(AudioEntry(audio_path=audio_path, audio_id=audio_path.stem))
        else:
            raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")
    else:
        audio_dir = audio_dir.resolve()
        for ext in suffixes:
            for path in sorted(audio_dir.rglob(f"*{ext}")):
                entries.append(AudioEntry(audio_path=path, audio_id=path.stem))

    if limit is not None:
        entries = entries[:limit]

    missing = [str(e.audio_path) for e in entries if not e.audio_path.exists()]
    if missing:
        raise FileNotFoundError(f"Found {len(missing)} missing audio files. Example: {missing[:3]}")

    LOGGER.info("Loaded %d audio entries", len(entries))
    return entries


def prepare_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> ASE:
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)

    model = ASE(config)
    # PyTorch >= 2.6 defaults weights_only=True; ensure compatibility with older checkpoints
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch without weights_only argument
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    # Filter out unexpected keys (e.g., HF buffers like embeddings.position_ids) for compatibility
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


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    with args.config.open("r") as handle:
        config = yaml.safe_load(handle)

    checkpoint_path = args.checkpoint or Path(config.get("eval", {}).get("ckpt", ""))
    if not checkpoint_path:
        raise ValueError("Checkpoint path must be provided via --checkpoint or config eval.ckpt.")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device_str = (
        args.device
        or config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    target_sr = int(config.get("audio_args", {}).get("sr", 32000))
    config_max_len = config.get("audio_args", {}).get("max_length", 0)
    max_length = None
    if args.max_duration is not None:
        max_length = int(args.max_duration * target_sr)
    elif config_max_len:
        max_length = int(config_max_len * target_sr)

    entries = load_manifest(
        manifest_path=args.manifest,
        audio_dir=args.audio_dir,
        audio_col=args.audio_col,
        id_col=args.id_col,
        suffixes=args.suffixes,
        limit=args.limit,
    )

    dataset = AudioDataset(
        entries=entries,
        target_sr=target_sr,
        max_length=max_length,
        mono=args.mono,
        deterministic=args.deterministic,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_npz = args.out_npz or (args.output_dir / "audio_embeddings.npz")
    out_parquet = args.out_parquet or (args.output_dir / "audio_embeddings.parquet")
    spec_json = args.spec_json or (args.output_dir / "embedding_spec.json")

    if args.dry_run:
        LOGGER.info("Dry-run complete. Would have processed %d files.", len(entries))
        return

    model = prepare_model(args.config, Path(checkpoint_path), device)

    save_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.save_dtype]

    clip_embeds: List[np.ndarray] = []
    audio_ids: List[str] = []
    audio_paths: List[str] = []

    with torch.inference_mode():
        for batch_waveforms, _, batch_meta in tqdm(dataloader, desc="Encoding", unit="batch"):
            batch_waveforms = batch_waveforms.to(device)

            _, frame_embeds = model.encode_audio(batch_waveforms)
            clip_batch = model.msc(frame_embeds, model.codebook)
            if args.normalize:
                frame_embeds = F.normalize(frame_embeds, dim=-1)
                clip_batch = F.normalize(clip_batch, dim=-1)

            clip_batch = clip_batch.to(torch.float32).cpu()
            for idx, item in enumerate(batch_meta):
                clip_embeds.append(clip_batch[idx].numpy().astype(args.save_dtype, copy=False))
                audio_ids.append(item["audio_id"])
                audio_paths.append(item["audio_path"])

    if clip_embeds:
        clip_matrix = np.stack(clip_embeds, axis=0)
    else:
        embed_dim = int(config.get("embed_size", 0))
        clip_matrix = np.empty((0, embed_dim), dtype=np.float32)

    np.savez(out_npz, audio_ids=np.array(audio_ids, dtype=object), audio_paths=np.array(audio_paths, dtype=object), clip_embeddings=clip_matrix)
    LOGGER.info("Wrote clip embeddings to %s", out_npz)

    if pd is not None:
        try:
            df = pd.DataFrame({
                "audio_id": audio_ids,
                "audio_path": audio_paths,
                "clip_embedding": list(clip_embeds),
            })
            df.to_parquet(out_parquet, index=False)
            LOGGER.info("Wrote clip embeddings to %s", out_parquet)
        except Exception as exc:  # pragma: no cover - optional dependency path
            LOGGER.warning("Failed to write Parquet file due to %s", exc)
    else:  # pragma: no cover - optional dependency path
        LOGGER.warning("pandas not available; skipping Parquet export")

    max_audio_seconds = (max_length / target_sr) if (max_length and target_sr) else None
    spec = {
        "model": str(checkpoint_path),
        "config": str(args.config),
        "device": device_str,
        "target_sample_rate": target_sr,
        "max_audio_length_samples": max_length,
        "max_audio_length_seconds": max_audio_seconds,
        "embedding_dim": int(clip_matrix.shape[1]) if clip_matrix.size else None,
        "pooling": "msc",
        "normalize": args.normalize,
        "save_dtype": args.save_dtype,
        "num_items": len(entries),
        "audio_column": args.audio_col,
        "id_column": args.id_col,
    }

    with spec_json.open("w") as handle:
        json.dump(spec, handle, indent=2)
    LOGGER.info("Wrote embedding spec to %s", spec_json)


if __name__ == "__main__":
    main()
