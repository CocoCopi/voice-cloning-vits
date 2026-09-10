"""Launch speaker-conditioned VITS training (YourTTS recipe) on a T4.

Thin wrapper over Coqui's trainer (same flow as TTS.bin.train_tts) plus the
three pipeline-specific steps:
  * registers the custom `libritts_subset` formatter BEFORE samples load
  * attaches the FROZEN pretrained speaker encoder to the speaker manager so
    the Speaker Consistency Loss (SCL) can run (Coqui never loads it from
    config paths automatically - verified in Vits.init_multispeaker)
  * rebuilds the audio resample transform (22050 -> 16000) that Vits only
    creates when `speaker_manager.encoder.audio_config` exists at __init__

Resume behavior: run again with the same config/output dir and it continues
from the latest checkpoint (Coqui trainer semantics via --continue_path).
"""

from __future__ import annotations

import argparse
import logging

import torch

logger = logging.getLogger(__name__)


def attach_frozen_speaker_encoder(model) -> None:
    """Attach the frozen pretrained SE to the model's EXISTING speaker manager.

    The speaker manager built by init_from_config already carries the d-vector
    embeddings that format_batch() consumes - replacing it would silently drop
    speaker conditioning. So we init the encoder IN PLACE, freeze it, and
    re-run init_multispeaker so the SCL branch + resample transform activate.
    """
    args = model.config.model_args
    if not args.use_speaker_encoder_as_loss:
        logger.info("SCL disabled - no speaker encoder attach needed")
        return
    if not args.speaker_encoder_model_path or not args.speaker_encoder_config_path:
        raise RuntimeError("SCL enabled but speaker_encoder_model_path/config_path missing")
    if model.speaker_manager is None:
        raise RuntimeError("model.speaker_manager is None; d-vectors must be configured")

    # init_encoder() sets .encoder/.encoder_ap/.encoder_config and keeps embeddings
    model.speaker_manager.init_encoder(
        args.speaker_encoder_model_path, args.speaker_encoder_config_path, use_cuda=False
    )
    model.speaker_manager.encoder.eval()
    for p in model.speaker_manager.encoder.parameters():
        p.requires_grad_(False)

    # init_multispeaker builds the 22050->16000 resample transform only when
    # the encoder exposes audio_config; provide it, then re-run to wire SCL.
    model.speaker_manager.encoder.audio_config = {"sample_rate": 16000}
    model.init_multispeaker(model.config)

    import torchaudio

    if model.audio_transform is None:
        model.audio_transform = torchaudio.transforms.Resample(
            orig_freq=model.config.audio.sample_rate, new_freq=16000
        ).to(model.device)
    else:
        model.audio_transform = model.audio_transform.to(model.device)

    n_frozen = sum(1 for p in model.speaker_manager.encoder.parameters() if not p.requires_grad)
    logger.info(
        "Frozen speaker encoder attached for SCL: %d frozen params, embeddings=%d clips, resample 22050->16000",
        n_frozen,
        model.speaker_manager.num_embeddings,
    )
    assert model.speaker_manager.num_embeddings > 0, "d-vector embeddings lost - aborting"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to vits_yourtts_*.json")
    ap.add_argument("--continue-path", default="", help="experiment folder to resume")
    ap.add_argument("--restore-path", default="", help="checkpoint to warm-start from")
    args = ap.parse_args()

    # 1. custom formatter must exist before load_tts_samples runs
    import build_config  # noqa: F401  (import registers libritts_subset formatter)

    # 2. load config + samples
    from TTS.config import load_config
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits

    config = load_config(args.config)
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    logger.info("train=%d eval=%d samples", len(train_samples), len(eval_samples))

    # 3. build model exactly like the verified d-vector tests:
    #    init_from_config -> SpeakerManager from use_d_vector_file -> init_multispeaker
    model = Vits.init_from_config(config)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    attach_frozen_speaker_encoder(model)

    # 4. trainer
    from trainer import Trainer, TrainerArgs

    train_args = TrainerArgs(
        continue_path=args.continue_path,
        restore_path=args.restore_path,
    )
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
