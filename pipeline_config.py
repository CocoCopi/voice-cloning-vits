"""Shared paths, constants and helpers for the voice-cloning pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout (overridable via env vars; Colab sets DATA_ROOT / OUTPUT_ROOT)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "output"))

VCTK_DIR = Path(os.environ.get("VCTK_DIR", DATA_ROOT / "VCTK-0.92"))
LIBRITTS_DIR = Path(os.environ.get("LIBRITTS_DIR", DATA_ROOT / "LibriTTS"))
DRIVE_CKPT_DIR = Path(os.environ.get("DRIVE_CKPT_DIR", PROJECT_ROOT / "drive_checkpoints"))

# Model / assets
SE_MODEL_PATH = Path(
    os.environ.get(
        "SE_MODEL_PATH",
        PROJECT_ROOT / "assets" / "speaker_encoder_model.ckpt",
    )
)
SE_CONFIG_PATH = Path(
    os.environ.get(
        "SE_CONFIG_PATH",
        PROJECT_ROOT / "configs" / "speaker_encoder_ecapa_16k.yaml",
    )
)
PHONEME_CACHE = Path(os.environ.get("PHONEME_CACHE", DATA_ROOT / "phoneme_cache"))

# 16 kHz mono - everything the frozen SE touches (d-vectors, SCL, eval) uses this
SE_SAMPLE_RATE = 16_000

# ---------------------------------------------------------------------------
# Speaker holdout
# ---------------------------------------------------------------------------

# Fixed seed => a speaker is held out from EVERY stage (training samples,
# d-vectors, SCL) consistently across sessions.
HOLDOUT_SEED = 1234

# ---------------------------------------------------------------------------
# ECAPA speaker-encoder YAML template (Coqui/BaseEncoder format).
# Mirrors Coqui's official ECAPA checkpoint (VoxCeleb, 512-d, L2-normalized).
# ---------------------------------------------------------------------------

SE_CONFIG_TEMPLATE = """\
model: "speaker_encoder"
model_name: "ecapa"
audio: {{
    fft_size: 512
    win_length: 400
    hop_length: 160
    frame_shift_ms: null
    frame_length_ms: null
    stft_pad_mode: "reflect"
    sample_rate: 16000
    resample: false
    preemphasis: 0.97
    ref_level_db: 20
    do_sound_norm: false
    log_func: "np.log10"
    do_trim_silence: true
    trim_db: 60
    do_rms_norm: false
    db_level: null
    power: 1.5
    griffin_lim_iters: 60
    num_mels: 80
    mel_fmin: 0.0
    mel_fmax: 8000.0
    spec_gain: 20.0
    do_amp_to_db_linear: true
    do_amp_to_db_mel: true
    pitch_fmax: 640.0
    pitch_fmin: 1.0
    signal_norm: true
    min_level_db: -100
    symmetric_norm: true
    max_norm: 4.0
    clip_norm: true
    stats_path: null
}}
model_params: {{
    input_dim: 80
    use_torch_spec: true
    log_input_audio: false
    proj_dim: 512
    channels: [1024, 1024, 1024, 1024, 3072]
    kernel_sizes: [5, 3, 3, 3, 1]
    dilations: [1, 2, 3, 4, 1]
    attention_channels: 128
    res2net_scale: 8
    se_channels: 128
    global_context: false
    grad_clip: null
}}
"""


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for p in (
        PROJECT_ROOT,
        DATA_ROOT,
        OUTPUT_ROOT,
        VCTK_DIR,
        LIBRITTS_DIR,
        DRIVE_CKPT_DIR,
        PHONEME_CACHE,
        SE_MODEL_PATH.parent,
        SE_CONFIG_PATH.parent,
    ):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset descriptor for run bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    """High-level description of one training configuration."""

    name: str = "vctk_yourtts_v1"
    dataset: str = "vctk"  # 'vctk' | 'libritts'
    heldout_speakers: list[str] = field(default_factory=list)
    num_train_speakers: int = 0
    notes: str = ""
