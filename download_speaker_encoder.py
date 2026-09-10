"""Download + verify the frozen pretrained speaker encoder (ECAPA-TDNN, VoxCeleb).

The speaker encoder is a SOLVED component (trained on ~1M+ VoxCeleb utterances).
We reuse it frozen for three jobs and never train it:
  1. d-vector computation for VITS training      (compute_embeddings.py)
  2. Speaker Consistency Loss (SCL) during VITS training (baked into the trainer)
  3. Held-out speaker similarity scoring          (evaluate_cloning.py)
  4. Reference-clip embedding at inference        (infer.py)

We deliberately use Coqui's ECAPA-TDNN checkpoint rather than SpeechBrain:
it shares the `BaseEncoder`/`AudioProcessor` code path with Coqui TTS, so the
exact same embedding pipeline powers training-time SCL, embedding computation,
eval and inference with zero glue code.

Usage:
    python download_speaker_encoder.py            # fetch + verify + write config
"""

from __future__ import annotations

import sys

import requests
from tqdm import tqdm

from pipeline_config import SE_CONFIG_PATH, SE_CONFIG_TEMPLATE, SE_MODEL_PATH

SE_URL = "https://github.com/coqui-ai/TTS/releases/download/v0.14.0_models/speaker_encoder_model.pth"


def download_speaker_encoder(force: bool = False) -> bool:
    """Download the pretrained ECAPA speaker encoder; returns True when present."""
    if SE_MODEL_PATH.exists() and SE_MODEL_PATH.stat().st_size > 50_000_000 and not force:
        print(f"[ok] speaker encoder already present: {SE_MODEL_PATH}")
        return True

    SE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dl] speaker encoder <- {SE_URL}")
    tmp_path = SE_MODEL_PATH.with_suffix(".tmp")
    with requests.get(SE_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="ecapa") as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
    tmp_path.rename(SE_MODEL_PATH)
    print(f"[ok] saved -> {SE_MODEL_PATH} ({SE_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    # Coqui checkpoints embed a `version` key (informational fingerprint).
    try:
        import torch

        ckpt = torch.load(SE_MODEL_PATH, map_location="cpu", weights_only=False)
        version = ckpt.get("version", "") if isinstance(ckpt, dict) else ""
        print(f"[info] checkpoint version tag: {version or '(none)'}")
    except Exception as exc:  # pragma: no cover - verification is best-effort
        print(f"[warn] could not inspect checkpoint: {exc}")
    return True


def write_se_config() -> None:
    """Write the SE audio/model config matching Coqui's ECAPA checkpoint."""
    SE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SE_CONFIG_PATH.write_text(SE_CONFIG_TEMPLATE)
    print(f"[ok] wrote SE config -> {SE_CONFIG_PATH}")


def verify_encoder_loads() -> None:
    """Load the encoder via Coqui's SpeakerManager and embed a synthetic clip.

    Catches wrong weights/config pairings BEFORE a long training run.
    """
    try:
        import numpy as np
        import torch

        from TTS.tts.utils.speakers import SpeakerManager
    except ImportError as exc:
        print(f"[skip] verification requires coqui-tts installed ({exc})")
        return

    mgr = SpeakerManager(encoder_model_path=SE_MODEL_PATH, encoder_config_path=SE_CONFIG_PATH, use_cuda=False)
    wav = (0.1 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000)).astype(np.float32)
    tmp = SE_MODEL_PATH.parent / "_verify_clip.wav"
    try:
        import soundfile as sf

        sf.write(tmp, wav, 16_000)
        emb = mgr.compute_embedding_from_clip(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    emb = list(emb)
    assert len(emb) == 512, f"expected 512-d embedding, got {len(emb)}"
    norm = float(np.linalg.norm(emb))
    assert 0.9 < norm < 1.1, f"embedding not L2-normalized (norm={norm:.3f})"
    print(f"[ok] encoder loads and embeds; 512-d, L2-norm={norm:.3f}")


def main() -> None:
    force = "--force" in sys.argv
    download_speaker_encoder(force=force)
    write_se_config()
    verify_encoder_loads()


if __name__ == "__main__":
    main()
