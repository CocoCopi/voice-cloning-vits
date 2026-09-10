"""Compute per-utterance speaker embeddings (d-vectors) with the frozen SE.

Produces a JSON mapping EVERY training/eval clip to its 512-d embedding:

    { "<dataset_name>#<relpath-without-ext>": {"name": "<speaker>", "embedding": [...] }, ... }

The key MUST be exactly `f"{dataset_name}#{relpath_without_ext}"` (relative to the
dataset root) because Coqui's `add_extra_keys` builds `audio_unique_name` in that
format and `Vits.format_batch` looks up d-vectors by that key. A mismatch here
fails training with a KeyError on the first batch.

Speakers listed in `heldout_speakers.json` are EXCLUDED here so they never leak
into training conditioning, while eval embeddings get their own file for scoring.

Usage:
    python compute_embeddings.py --dataset vctk
    python compute_embeddings.py --dataset libritts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from pipeline_config import (
    SE_CONFIG_PATH,
    SE_MODEL_PATH,
    VCTK_DIR,
    LIBRITTS_DIR,
)
from speaker_logic import raw_speaker_from_path, rel_key, speaker_from_path


def collect_wavs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".flac", ".wav"})


def compute_split(
    manager,
    wavs: list[Path],
    root: Path,
    dataset_name: str,
    speaker_of: callable,
    out_path: Path,
) -> None:
    embeddings: dict[str, dict] = {}
    errors = 0
    for wav in tqdm(wavs, desc=f"embed {out_path.stem}", unit="utt"):
        key = rel_key(wav, root, dataset_name)
        try:
            emb = manager.compute_embedding_from_clip(str(wav))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            tqdm.write(f"[skip] {wav.name}: {exc}")
            continue
        embeddings[key] = {"name": speaker_of(wav), "embedding": [float(x) for x in emb]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    print(f"[ok] {len(embeddings)} embeddings -> {out_path} ({errors} skipped)")
    if errors > len(wavs) * 0.05:
        print("[error] >5% of clips failed to embed; investigate before training")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vctk", "libritts", "both"], default="vctk")
    ap.add_argument("--batch-limit", type=int, default=0, help="debug: cap number of clips")
    args = ap.parse_args()

    from TTS.tts.utils.speakers import SpeakerManager

    manager = SpeakerManager(
        encoder_model_path=SE_MODEL_PATH,
        encoder_config_path=SE_CONFIG_PATH,
        use_cuda=torch.cuda.is_available(),
    )

    for ds in ["vctk", "libritts"] if args.dataset == "both" else [args.dataset]:
        root = VCTK_DIR if ds == "vctk" else LIBRITTS_DIR
        wavs = collect_wavs(root)
        if args.batch_limit:
            wavs = wavs[: args.batch_limit]
        if not wavs:
            print(f"[error] no audio found under {root} - run the prep script first")
            sys.exit(1)

        # Holdout speakers are excluded from the TRAIN file (never conditioned on),
        # but still embedded into a separate eval file used for scoring.
        heldout_path = root / "metadata" / "heldout_speakers.json"
        heldout = set(json.loads(heldout_path.read_text())) if heldout_path.exists() else set()

        train_wavs = [w for w in wavs if raw_speaker_from_path(w) not in heldout]
        eval_wavs = [w for w in wavs if raw_speaker_from_path(w) in heldout] if heldout else []

        compute_split(
            manager,
            train_wavs,
            root,
            ds,
            speaker_from_path,
            root / "metadata" / "d_vectors_train.json",
        )
        if eval_wavs:
            compute_split(
                manager,
                eval_wavs,
                root,
                ds,
                speaker_from_path,
                root / "metadata" / "d_vectors_eval.json",
            )


if __name__ == "__main__":
    main()
