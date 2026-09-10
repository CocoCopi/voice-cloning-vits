# Zero-Shot Voice Cloning — Speaker-Conditioned VITS (YourTTS recipe)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CocoCopi/voice-cloning-vits/blob/main/voice_cloning_colab.ipynb)

Train **your own** text-to-speech model — not a wrapper around someone else's API — that takes any
short reference clip at inference and clones that voice, generalizing to speakers never seen during
training. Designed to train on a single T4 (16 GB) across multiple Colab sessions.

## Why this generalizes

The model is a **speaker-conditioned VITS** (the YourTTS design):

1. One VITS decoder trained over many speakers simultaneously, conditioned on 512-d speaker
   embeddings (d-vectors) computed by a **frozen, pretrained** ECAPA-TDNN speaker encoder.
2. Because conditioning is an *embedding space* — not a fixed table of voices — the decoder learns
   the concept "apply this voice's characteristics" and transfers it to brand-new embeddings at
   inference. Generalization comes from **speaker diversity in training data**, not model size.
3. The Speaker Consistency Loss (SCL) runs the same frozen encoder over generated speech during
   training, pulling synthetic embeddings toward the reference.
4. The speaker encoder itself is never trained — speaker verification is a solved problem
   (VoxCeleb-scale); we only train the TTS decoder to *use* those embeddings.

## Repository layout

```
configs/speaker_encoder_ecapa_16k.yaml  # frozen SE config (ECAPA, 16 kHz, 512-d)
download_speaker_encoder.py             # fetch + verify the pretrained ECAPA checkpoint
prepare_vctk.py                         # VCTK 0.92 -> formatter layout, 22050 Hz mono
prepare_libritts.py                     # LibriTTS subset: speaker-capped, holdout split
compute_embeddings.py                   # per-utterance d-vectors (frozen SE) -> JSON
build_config.py                         # YourTTS-style VITS config + custom formatter
train_tts_model.py                      # training launch (fp16, resume, SCL wiring)
evaluate_cloning.py                     # held-out speaker similarity + UTMOS naturalness
infer.py                                # reference clip(s) + text -> cloned speech
tts_runtime.py                          # shared checkpoint loading / synthesis
speaker_logic.py                        # torch-free speaker/d-vector key helpers
tests/test_pipeline_logic.py            # pins the fragile contracts (d-vector keys, splits)
audio_utils.py                          # corpus resampling to 22050 Hz mono
build_colab_notebook.py                 # regenerates voice_cloning_colab.ipynb
```

Run the contract tests after touching key/parsing logic: `pytest tests/ -q`

## Quickstart (Colab, T4)

Open the notebook via the **Open in Colab** badge above (or from
[github.com/CocoCopi/voice-cloning-vits](https://github.com/CocoCopi/voice-cloning-vits)) and run
top-to-bottom. The repo is public, so Colab clones it without any auth. The notebook:

1. mounts Drive (datasets + checkpoints survive disconnects),
2. installs `coqui-tts` (the maintained idiap fork) on Colab's CUDA torch,
3. downloads/verifies the frozen ECAPA speaker encoder,
4. **Phase A**: prepares VCTK, computes d-vectors, builds the config, trains (auto-resume),
5. **Phase B**: repeats on a speaker-capped LibriTTS subset (~2.4k speakers) for real generalization,
6. evaluates held-out cloning, and clones from your own uploaded clip.

Regenerate the notebook after editing cells:

```bash
python build_colab_notebook.py
```

## Quickstart (local / scripts)

```bash
pip install -r requirements.txt          # after a CUDA-matched torch/torchaudio

python download_speaker_encoder.py
python prepare_vctk.py                   # or: --from-zip VCTK-Corpus-0.92.zip
python compute_embeddings.py --dataset vctk

python build_config.py --dataset vctk \
  --d-vectors data/VCTK-0.92/metadata/d_vectors_train.json \
  --heldout   data/VCTK-0.92/metadata/heldout_speakers.json \
  --batch-size 16 --grad-accum 2

python train_tts_model.py --config configs/vits_yourtts_vctk.json   # re-run to resume

python evaluate_cloning.py --model <ckpt.pth> --config configs/vits_yourtts_vctk.json
python infer.py --model <ckpt.pth> --config configs/vits_yourtts_vctk.json \
  --ref ref.wav --text "Hello from a cloned voice."
```

## The generalization test (do not skip)

`prepare_*` scripts hold out test speakers **entirely** — they never appear in training samples,
d-vectors, or SCL (enforced in `compute_embeddings.py` and the formatter ignore lists). After
training, `evaluate_cloning.py` measures:

- **Speaker similarity** — cosine between the frozen ECAPA embedding of generated audio and the
  mean reference embedding, for held-out speakers, training-speaker controls, and a random-embedding
  baseline (~0.0).
- **Naturalness** — UTMOS predicted MOS when available.

| held-out cosine | meaning |
|---|---|
| ≥ 0.65 | recognizable clone |
| 0.50–0.65 | same voice region; add speakers / continue training |
| ≤ 0.45 | not cloning (compare to baseline) |

**Decision rule**: held-out weak + training-speakers strong ⇒ the model memorized; **add speakers**
(Phase B), not epochs on the same data.

## T4 budget notes

- fp16 (`mixed_precision`), batch 16 × grad-accum 2 (effective 32), `max_audio_len` 12 s.
- Checkpoints every 2500 steps to Drive; `save_n_checkpoints: 4`; Ctrl-C saves on interrupt.
- Phase A (VCTK): first audible samples ~30–50k steps; overnight run gives a real signal.
- Phase B (LibriTTS, 10 utt/spk): multi-session; every session re-runs the same train cell and it
  resumes from the latest checkpoint.

## Gotchas baked into the code (why it's built this way)

- **d-vector keys**: Coqui looks up embeddings by `dataset#relative-path-without-extension`
  (`add_extra_keys`); `compute_embeddings.py` emits exactly that format or training KeyErrors.
- **Sample rates**: Coqui's `VitsDataset` never resamples; all corpora are normalized to 22050 Hz
  mono by `audio_utils.py`, and the SCL path resamples to the SE's 16 kHz on-GPU.
- **Frozen SE attach**: Coqui never loads the SE from config paths by itself; `train_tts_model.py`
  attaches it to the *existing* speaker manager (preserving d-vectors) and rebuilds the resampler.
- **Speaker-balanced sampling** (`use_weighted_sampler` on `speaker_name`) so frequent speakers
  don't dominate the conditioning space.
