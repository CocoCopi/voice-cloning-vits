"""Generates voice_cloning_colab.ipynb (T4 end-to-end runbook).

Run:  python build_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("voice_cloning_colab.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.append(
    md(
        """# 🎙️ Zero-Shot Voice Cloning — Speaker-Conditioned VITS (YourTTS recipe)

Train **your own** TTS model (no third-party inference API) that clones **any** voice from a short
reference clip — including speakers never seen during training.

**The mechanism** (why this generalizes): one VITS decoder is trained over MANY speakers at once,
conditioned on 512-d speaker embeddings (d-vectors) from a **frozen, pretrained** ECAPA-TDNN speaker
encoder. Because the decoder must *use* embeddings of hundreds of voices, it learns the concept of
"apply this voice" — which transfers to brand-new embeddings at inference.

**Pipeline on this notebook**

| Stage | Script | Notes |
|---|---|---|
| 1. Frozen speaker encoder | `download_speaker_encoder.py` | ECAPA-TDNN (VoxCeleb), never trained |
| 2. Data | `prepare_vctk.py` → `prepare_libritts.py` | VCTK first (validate), LibriTTS for real diversity |
| 3. d-vectors | `compute_embeddings.py` | one embedding per utterance (training conditioning) |
| 4. Config | `build_config.py` | YourTTS-style VITS + speaker-consistency loss (SCL) |
| 5. Training | `train_tts_model.py` | fp16, grad-accum, resume-safe, checkpoints to Drive |
| 6. **Held-out eval** | `evaluate_cloning.py` | ECAPA cosine similarity + UTMOS — the real finish line |
| 7. Inference | `infer.py` | any clip + text → cloned speech |

**Runtime**: T4 (16 GB). Phase A (VCTK validation): ~1 day of training for a first signal.
Phase B (LibriTTS subset): multi-session — checkpoints auto-resume.

**Rule of thumb**: if held-out cloning is weak but training-speaker cloning is strong →
add MORE SPEAKERS (Phase B), not more epochs.
"""
    )
)

cells.append(
    code(
        """# ── 1. GPU check + Drive mount ────────────────────────────────────────────────
import torch, sys
assert torch.cuda.is_available(), "Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0), "| VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

from google.colab import drive  # type: ignore
drive.mount('/content/drive')

import os
# Everything heavy lives on Drive so it survives disconnects:
os.environ['PROJECT_ROOT'] = '/content/drive/MyDrive/voice_clone'
os.environ['DATA_ROOT']    = '/content/drive/MyDrive/voice_clone/data'
os.environ['OUTPUT_ROOT']  = '/content/drive/MyDrive/voice_clone/output'
for k in ('PROJECT_ROOT', 'DATA_ROOT', 'OUTPUT_ROOT'):
    os.makedirs(os.environ[k], exist_ok=True)
print("data ->", os.environ['DATA_ROOT'])
"""
    )
)

cells.append(
    code(
        """# ── 2. Get the pipeline code (public repo - no auth needed) ─────────────
import os
REPO_DIR = "/content/voice-cloning-vits"

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/CocoCopi/voice-cloning-vits.git {REPO_DIR}
%cd {REPO_DIR}
!git pull || true

# Datasets/checkpoints live on Drive; code runs from the (fast) local clone.
import pathlib
pathlib.Path('/content/drive/MyDrive/voice_clone/data').mkdir(parents=True, exist_ok=True)
!ln -sfn /content/drive/MyDrive/voice_clone/data {REPO_DIR}/data
"""
    )
)

cells.append(
    code(
        """# ── 3. Install stack (keep Colab's preinstalled CUDA torch!) ────────────────
# Do NOT reinstall torch here - Colab's build matches its CUDA runtime and
# coqui-tts only requires torch>=2.2, which is already satisfied.
!pip install -q 'coqui-tts[cuda]>=0.27.0,<0.28' speechbrain jiwer gdown

import torch, torchaudio, TTS
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__, "| TTS", TTS.__version__)
from TTS.tts.configs.vits_config import VitsConfig  # import smoke test
from TTS.tts.datasets.formatters import register_formatter  # custom formatter hooks exist
print("VitsConfig import OK")
"""
    )
)

cells.append(
    code(
        """# ── 4. Frozen pretrained speaker encoder (ECAPA-TDNN, VoxCeleb) ─────────────
# Solved component - we reuse it, never train it. Feeds d-vectors for training,
# the Speaker Consistency Loss during training, eval scoring, and inference.
!python download_speaker_encoder.py
"""
    )
)

cells.append(
    md(
        """## Phase A — VCTK (pipeline validation, ~109 speakers)

Goal: prove every piece end-to-end. Generalization will be modest (109 speakers) — that's expected;
Phase B is where cloning quality jumps.
"""
    )
)

cells.append(
    code(
        """# ── 5. Download + prepare VCTK 0.92 (~30-45 min) ────────────────────────────
!python prepare_vctk.py
"""
    )
)

cells.append(
    code(
        """# ── 6. Hold out test speakers (NEVER trained on) ────────────────────────────
# Fixed list => the same speakers stay held out across sessions and phases.
import json, os, pathlib
meta_dir = pathlib.Path(os.environ['DATA_ROOT']) / 'VCTK-0.92' / 'metadata'
meta_dir.mkdir(parents=True, exist_ok=True)
heldout = ["p239", "p245", "p326", "p334", "p362"]  # 5 unseen test voices
(meta_dir / 'heldout_speakers.json').write_text(json.dumps(heldout, indent=2))
print("held out:", heldout)
"""
    )
)

cells.append(
    code(
        """# ── 7. Compute d-vectors for every training clip (~20-40 min on GPU) ────────
!python compute_embeddings.py --dataset vctk
"""
    )
)

cells.append(
    code(
        """# ── 8. Build the YourTTS-style config ───────────────────────────────────────
!python build_config.py --dataset vctk \\
    --d-vectors data/VCTK-0.92/metadata/d_vectors_train.json \\
    --heldout   data/VCTK-0.92/metadata/heldout_speakers.json \\
    --batch-size 16 --grad-accum 2 --epochs 500
"""
    )
)

cells.append(
    md(
        """## Training (resumable)

- fp16 + grad-accum → fits 16 GB. Batch 16 × accum 2 = effective 32.
- Checkpoints every 2500 steps → Drive (`output_path` is on Drive).
- **Reconnect/disconnect? Just re-run this cell** — it finds the latest checkpoint and continues.
- VCTK signal: loss drops steadily; listen to TensorBoard audio after ~50k steps.
"""
    )
)

cells.append(
    code(
        """# ── 9. Train (auto-resumes) ──────────────────────────────────────────────────
import glob, os, sys

CFG = "configs/vits_yourtts_vctk.json"
exp_root = os.path.join(os.environ['OUTPUT_ROOT'], "vits_vctk")
ckpts = sorted(glob.glob(os.path.join(exp_root, "**", "checkpoint_*.pth"), recursive=True))

cmd = [sys.executable, "train_tts_model.py", "--config", CFG]
if ckpts:
    run_dir = os.path.dirname(ckpts[-1])
    print(f"[resume] continuing from {run_dir}")
    cmd += ["--continue-path", run_dir]
else:
    print("[fresh] no checkpoints found - starting a new run")

# Long-running: Colab keeps streaming stdout; interrupt Cell to stop safely
# (trainer saves on interrupt). Re-run this cell to resume.
!{' '.join(cmd)}
"""
    )
)

cells.append(
    code(
        """# ── 10. (optional) TensorBoard ────────────────────────────────────────────────
%load_ext tensorboard
%tensorboard --logdir {os.environ['OUTPUT_ROOT']}/vits_vctk
"""
    )
)

cells.append(
    md(
        """## Phase B — LibriTTS subset (the real generalization driver)

~2,456 speakers. Capping at ~10 utterances/speaker keeps the multi-session budget sane while
maximizing **speaker diversity** — the thing that actually produces "clones voices it has never
heard". Re-run steps 7–9 with the `libritts` config afterwards.
"""
    )
)

cells.append(
    code(
        """# ── 11. LibriTTS subset (download + speaker-capped selection) ────────────────
!python prepare_libritts.py --max-utts-per-spk 10          # add --with-360 for more
"""
    )
)

cells.append(
    code(
        """# ── 12. d-vectors + config for LibriTTS ─────────────────────────────────────
!python compute_embeddings.py --dataset libritts
!python build_config.py --dataset libritts \\
    --d-vectors data/LibriTTS/metadata/d_vectors_train.json \\
    --heldout   data/LibriTTS/metadata/heldout_speakers.json \\
    --batch-size 12 --grad-accum 3 --epochs 800
# then re-run cell 9 with CFG="configs/vits_yourtts_libritts.json"
"""
    )
)

cells.append(
    md(
        """## Held-out speaker evaluation — the real finish line

Scores **speakers the model never saw**: reference clips → ECAPA embedding → synthesis →
re-embed → cosine similarity (+ UTMOS naturalness). Includes training-speaker control and a
random-embedding baseline (~0.0) so the numbers mean something.
"""
    )
)

cells.append(
    code(
        """# ── 13. Evaluate cloning on held-out speakers ────────────────────────────────
import glob, os

def best_checkpoint(exp_root):
    best = sorted(glob.glob(os.path.join(exp_root, "**", "best_model*.pth"), recursive=True))
    if best:
        return best[-1]
    ckpts = sorted(glob.glob(os.path.join(exp_root, "**", "checkpoint_*.pth"), recursive=True))
    return ckpts[-1] if ckpts else None

exp_root = os.path.join(os.environ['OUTPUT_ROOT'], "vits_vctk")   # or ..._libritts
CKPT = best_checkpoint(exp_root)
print("evaluating:", CKPT)

!python evaluate_cloning.py --model {CKPT} --config configs/vits_yourtts_vctk.json --dataset vctk
# JSON report lands next to the checkpoint: cloning_eval.json
"""
    )
)

cells.append(
    md(
        """### Reading the numbers

| held-out cosine | meaning |
|---|---|
| ≥ 0.65 | recognizable clone |
| 0.50–0.65 | same-voice-region match; add speakers / train longer |
| ≤ 0.45 | not cloning (random baseline ≈ 0) |

**Train-speaker similarity ≫ held-out similarity ⇒ add speakers (Phase B), not epochs.**
"""
    )
)

cells.append(
    md(
        """## Inference — clone ANY voice
"""
    )
)

cells.append(
    code(
        """# ── 14. Clone from a brand-new reference clip ───────────────────────────────
from google.colab import files  # type: ignore
print("Upload a 5-10 s clean speech clip (wav/flac/mp3):")
up = files.upload()
ref = next(iter(up))
!python infer.py --model {CKPT} --config configs/vits_yourtts_vctk.json --ref {ref} --text "This voice was cloned from a clip of audio I uploaded just now." --out /content/cloned.wav

import soundfile as sf, IPython.display as ipd
wav, sr = sf.read('/content/cloned.wav')
ipd.Audio(wav, rate=sr)
"""
    )
)

cells.append(
    md(
        """## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM | lower `--batch-size` (raise `--grad-accum` to compensate), or lower `max_audio_len` in `build_config.py` |
| `KeyError` on d-vector key | d-vector JSON keys must be `dataset#relpath-without-ext`; regenerate with `compute_embeddings.py` |
| Metallic/unstable audio early | normal before ~30-50k steps; discriminator warms up late |
| Cloning flat (all voices similar) | SCL alpha too low / too few speakers → check `speaker_encoder_loss_alpha`, go Phase B |
| Session died | re-run cell 9; it resumes from the latest checkpoint |
"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")
