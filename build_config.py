"""Build the YourTTS-style speaker-conditioned VITS training config.

The zero-shot mechanism (from the YourTTS recipe):
  * ONE VITS model conditioned on 512-d speaker embeddings (d-vectors) computed
    by a FROZEN, pretrained speaker encoder (ECAPA-TDNN). At inference, any new
    reference clip is embedded by the same frozen encoder and fed as `g` to the
    flow/decoder - a speaker never seen in training included.
  * Speaker Consistency Loss (SCL): the same frozen encoder scores generated
    speech during training, pulling synthetic embeddings toward the reference.
  * Speaker-balanced sampling so no voice dominates training.

Writes `configs/vits_yourtts_<dataset>.json` for `train_tts_model.py`.

Usage:
    python build_config.py --dataset vctk \
        --d-vectors data/VCTK-0.92/metadata/d_vectors_train.json \
        --heldout data/VCTK-0.92/metadata/heldout_speakers.json \
        --output configs/vits_yourtts_vctk.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_config import (
    DRIVE_CKPT_DIR,
    PHONEME_CACHE,
    SE_CONFIG_PATH,
    SE_MODEL_PATH,
    SE_SAMPLE_RATE,
    VCTK_DIR,
    LIBRITTS_DIR,
)

VCTK_BAD = ["p280", "p315"]  # known VCTK session issues (mic2 missing / corrupted)


def build_config(
    dataset: str,
    d_vector_file: list[str],
    heldout_speakers: list[str],
    num_speakers: int,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    output_path: Path,
) -> dict:
    root = VCTK_DIR if dataset == "vctk" else LIBRITTS_DIR

    ignored = list(VCTK_BAD) if dataset == "vctk" else []
    if heldout_speakers:
        # prefix-aware ignore list for the formatter
        ignored += heldout_speakers if dataset == "vctk" else [f"LTTS_{s}" for s in heldout_speakers]

    config = {
        "model": "vits",
        "run_name": f"yourtts_style_{dataset}",
        "run_description": "Speaker-conditioned VITS (YourTTS recipe) with frozen ECAPA SE for zero-shot cloning",
        # ---- logging / checkpointing (survives Colab disconnects) ----
        "output_path": str(DRIVE_CKPT_DIR / f"vits_{dataset}"),
        "print_step": 50,
        "plot_step": 500,
        "save_step": 2500,          # frequent checkpoints: multi-session runs
        "save_n_checkpoints": 4,
        "save_checkpoints": True,
        "run_eval": True,
        "print_eval": True,
        "test_delay_epochs": 0,
        "epochs": epochs,
        # ---- T4 sizing: fp16 + grad accumulation ----
        "mixed_precision": True,
        "precision": "fp16",
        "grad_accum_steps": grad_accum,
        "batch_size": batch_size,
        "eval_batch_size": max(4, batch_size // 2),
        "num_loader_workers": 2,
        "num_eval_loader_workers": 2,
        "allow_tf32": True,
        "cudnn_benchmark": False,
        "training_seed": 42,
        # ---- audio: 22050 Hz, standard VITS mel setup ----
        "audio": {
            "fft_size": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "frame_shift_ms": None,
            "frame_length_ms": None,
            "stft_pad_mode": "reflect",
            "sample_rate": 22050,
            "resample": False,
            "preemphasis": 0.0,
            "ref_level_db": 20,
            "do_sound_norm": False,
            "log_func": "np.log10",
            "do_trim_silence": True,
            "trim_db": 45,
            "do_rms_norm": False,
            "db_level": None,
            "power": 1.5,
            "griffin_lim_iters": 60,
            "num_mels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "spec_gain": 20.0,
            "do_amp_to_db_linear": True,
            "do_amp_to_db_mel": True,
            "pitch_fmax": 640.0,
            "pitch_fmin": 1.0,
            "signal_norm": True,
            "min_level_db": -100,
            "symmetric_norm": True,
            "max_norm": 4.0,
            "clip_norm": True,
            "stats_path": None,
        },
        # ---- datasets (formatter + root + meta file) ----
        "datasets": [
            _dataset_entry(dataset, root, ignored)
        ],
        # ---- text: char-level, no external phonemizer dependency ----
        "characters": {
            "characters_class": "TTS.tts.utils.text.characters.GraphemeCharacters",
            "characters": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',.!?;: \"-",
            "punctuations": "!',.:;? ",
            "pad": "_",
            "eos": "~",
            "bos": "^",
            "blank": "@",
        },
        "text_cleaner": "english_cleaners",
        "use_phonemes": False,
        "phoneme_language": "en-us",
        # ---- model args: speaker conditioning + SCL ----
        "model_args": {
            "num_chars": 100,
            "out_channels": 513,
            "spec_segment_size": 32,
            "hidden_channels": 192,
            "hidden_channels_ffn_text_encoder": 768,
            "num_heads_text_encoder": 2,
            "num_layers_text_encoder": 6,
            "kernel_size_text_encoder": 3,
            "dropout_p_text_encoder": 0.1,
            "dropout_p_duration_predictor": 0.5,
            "kernel_size_posterior_encoder": 5,
            "dilation_rate_posterior_encoder": 1,
            "num_layers_posterior_encoder": 16,
            "kernel_size_flow": 5,
            "dilation_rate_flow": 1,
            "num_layers_flow": 4,
            "resblock_type_decoder": "1",
            "resblock_kernel_sizes_decoder": [3, 7, 11],
            "resblock_dilation_sizes_decoder": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates_decoder": [8, 8, 2, 2],
            "upsample_initial_channel_decoder": 512,
            "upsample_kernel_sizes_decoder": [16, 16, 4, 4],
            "periods_multi_period_discriminator": [2, 3, 5, 7, 11],
            "use_sdp": True,
            "noise_scale": 1.0,
            "inference_noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_scale_dp": 1.0,
            "inference_noise_scale_dp": 1.0,
            "max_inference_len": None,
            "init_discriminator": True,
            "use_spectral_norm_disriminator": False,
            # --- speaker conditioning (the cloning mechanism) ---
            "use_speaker_embedding": False,
            "num_speakers": num_speakers,
            "speaker_embedding_channels": 256,
            "use_d_vector_file": True,
            "d_vector_file": d_vector_file,
            "d_vector_dim": 512,
            # --- speaker consistency loss (SCL / SE-loss) ---
            "use_speaker_encoder_as_loss": True,
            "speaker_encoder_model_path": str(SE_MODEL_PATH),
            "speaker_encoder_config_path": str(SE_CONFIG_PATH),
            "condition_dp_on_speaker": True,
            "detach_dp_input": True,
            # --- language conditioning (off for English-only v1) ---
            "use_language_embedding": False,
            "embedded_language_dim": 4,
            "num_languages": 0,
            "language_ids_file": None,
            # --- freezing (all trainable; the SE is external + eval'd) ---
            "freeze_encoder": False,
            "freeze_DP": False,
            "freeze_PE": False,
            "freeze_flow_decoder": False,
            "freeze_waveform_decoder": False,
            "encoder_sample_rate": None,
            "interpolate_z": True,
        },
        # ---- optimization (VITS defaults, conservative on T4) ----
        "grad_clip": [1000.0, 1000.0],
        "lr_gen": 0.0002,
        "lr_disc": 0.0002,
        "lr_scheduler_gen": "ExponentialLR",
        "lr_scheduler_gen_params": {"gamma": 0.999875, "last_epoch": -1},
        "lr_scheduler_disc": "ExponentialLR",
        "lr_scheduler_disc_params": {"gamma": 0.999875, "last_epoch": -1},
        "scheduler_after_epoch": True,
        "optimizer": "AdamW",
        "optimizer_params": {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01},
        "kl_loss_alpha": 1.0,
        "disc_loss_alpha": 1.0,
        "gen_loss_alpha": 1.0,
        "feat_loss_alpha": 1.0,
        "mel_loss_alpha": 45.0,
        "dur_loss_alpha": 1.0,
        "speaker_encoder_loss_alpha": 1.0,
        # ---- data loader ----
        "return_wav": True,
        "compute_linear_spec": True,
        "r": 1,
        "add_blank": True,
        "min_text_len": 5,
        "max_text_len": 220,
        "min_audio_len": int(0.6 * 22050),
        "max_audio_len": int(12.0 * 22050),  # hard VRAM ceiling on T4
        "precompute_num_workers": 2,
        "batch_group_size": 16,   # mild bucket shuffling for stable GAN batches
        "eval_split_size": 0.02,
        "eval_split_max_size": 256,
        "shuffle": True,
        "drop_last": False,
        "use_weighted_sampler": True,          # speaker-balanced sampling
        "weighted_sampler_attrs": {"speaker_name": 1.0},
        "weighted_sampler_multipliers": {},
        "phoneme_cache_path": str(PHONEME_CACHE / dataset),
        "speakers_file": None,
        "language_ids_file": None,
        # ---- test sentences for TensorBoard audio samples ----
        "test_sentences": [
            ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."],
            ["Be a voice, not an echo."],
            ["I'm sorry Dave. I'm afraid I can't do that."],
        ],
    }
    # NOTE: no non-VitsConfig keys are embedded in the config (Coqpit may reject
    # unknown fields). Holdout speakers live in <dataset>/metadata/heldout_speakers.json.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[ok] config -> {output_path}")
    return config


def _dataset_entry(dataset: str, root: Path, ignored: list[str]) -> dict:
    if dataset == "vctk":
        return {
            "formatter": "vctk",
            "dataset_name": "vctk",
            "path": str(root),
            "meta_file_train": "",
            "ignored_speakers": ignored,
            "language": "en",
        }
    # LibriTTS subset: register a tiny custom formatter over our curated TSVs
    register_libritts_subset_formatter()
    return {
        "formatter": "libritts_subset",
        "dataset_name": "libritts",
        "path": str(root),
        "meta_file_train": "metadata/libritts_train.tsv",
        "ignored_speakers": ignored,
        "language": "en",
    }


_FORMATTER_REGISTERED = False


def register_libritts_subset_formatter() -> None:
    """Register a custom formatter for our curated LibriTTS subset TSVs.

    Uses Coqui's public `register_formatter` API; TSV rows are (relpath, text).
    Speaker name is parsed from the LibriTTS utt id: <spk>_<chapter>_<utt>.
    """
    global _FORMATTER_REGISTERED
    if _FORMATTER_REGISTERED:
        return

    from TTS.tts.datasets.formatters import register_formatter

    def libritts_subset(root_path, meta_file, ignored_speakers=None, **kwargs):
        import os

        items = []
        meta_path = os.path.join(root_path, meta_file)
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                wav_rel, text = line.rstrip("\n").split("\t", 1)
                spk = os.path.basename(wav_rel).split("_")[0]
                if ignored_speakers and spk in ignored_speakers:
                    continue
                wav_file = os.path.join(root_path, wav_rel)
                if not os.path.exists(wav_file):
                    continue
                items.append(
                    {"text": text, "audio_file": wav_file, "speaker_name": f"LTTS_{spk}", "root_path": root_path}
                )
        return items

    register_formatter("libritts_subset", libritts_subset)
    _FORMATTER_REGISTERED = True
    print("[ok] registered custom formatter 'libritts_subset'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vctk", "libritts"], default="vctk")
    ap.add_argument("--d-vectors", type=Path, required=True, help="d_vectors_train.json path")
    ap.add_argument("--heldout", type=Path, default=None, help="heldout_speakers.json path")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    d_vector_file = [str(args.d_vectors)]
    heldout = json.loads(args.heldout.read_text()) if args.heldout and args.heldout.exists() else []

    import torch  # noqa: F401 - ensure torch present before coqui imports
    from TTS.tts.configs.vits_config import VitsConfig

    # count unique speakers from the d-vector file itself (source of truth)
    with open(args.d_vectors) as f:
        dv = json.load(f)
    num_speakers = len({v["name"] for v in dv.values()})

    out = args.output or Path(f"configs/vits_yourtts_{args.dataset}.json")
    cfg = build_config(
        dataset=args.dataset,
        d_vector_file=d_vector_file,
        heldout_speakers=heldout,
        num_speakers=num_speakers,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        output_path=out,
    )
    # provenance lives beside the config, never inside it
    out.with_suffix(".provenance.json").write_text(
        json.dumps({"heldout_speakers": heldout, "num_train_speakers": num_speakers}, indent=2),
        encoding="utf-8",
    )

    # hard validation: the real VitsConfig must accept our JSON
    vits_cfg = VitsConfig(**cfg)
    if hasattr(vits_cfg, "check_values"):
        vits_cfg.check_values()
    print(f"[ok] VitsConfig validated; {num_speakers} train speakers, {len(heldout)} held out")


if __name__ == "__main__":
    main()
