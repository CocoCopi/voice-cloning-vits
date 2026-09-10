"""Shared model loading + cloning synthesis used by eval and inference.

Both entrypoints need the same two things:
  * a VITS checkpoint restored with the frozen SE attached to the speaker
    manager (Coqui only auto-attaches it during training init)
  * reference-clip cloning via `model.synthesize(..., speaker_wav=...)`,
    which routes through BaseTTS._get_speaker_id_or_dvector -> clone_voice
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pipeline_config import SE_CONFIG_PATH, SE_MODEL_PATH


def load_model(config_path: str | Path, ckpt_path: str | Path):
    """Restore a trained VITS model ready for zero-shot cloning inference."""
    from TTS.config import load_config
    from TTS.tts.models.vits import Vits

    config = load_config(str(config_path))
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, str(ckpt_path), eval=True)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # inference-time cloning requires the encoder attached to the speaker manager
    model.speaker_manager.init_encoder(
        str(SE_MODEL_PATH), str(SE_CONFIG_PATH), use_cuda=torch.cuda.is_available()
    )
    return model, config


def synthesize(model, text: str, ref_files: list[str | Path], language: str | None = "en") -> np.ndarray:
    """Clone `text` in the voice of `ref_files` (one or more reference clips)."""
    out = model.synthesize(
        text=text,
        speaker_wav=[str(f) for f in ref_files],
        language=language,
    )
    return np.asarray(out["wav"], dtype=np.float32)
