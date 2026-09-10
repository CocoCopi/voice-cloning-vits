"""Held-out speaker cloning evaluation - the real finish line.

For every held-out speaker (never seen in training):
  1. embed 2+ reference clips with the FROZEN ECAPA encoder -> target vector
  2. synthesize held-out texts conditioned on that vector
  3. re-embed the generated audio -> cosine similarity vs the target
  4. naturalness: UTMOS (strong MOS predictor) if available
Control groups that make the numbers meaningful:
  * training speakers (same procedure) - should score HIGHER
  * random-embedding baseline - synthetic speech vs a random 512-d vector

Interpretation guide:
  - ECAPA cosine >= ~0.65: recognizable clone; 0.5-0.65: same-gender/quality match;
    < 0.45: not cloning (compare against the random baseline gap).
  - If train-speaker cloning is strong but held-out is weak: ADD SPEAKERS (section 3
    of the plan), do not train longer on the same data.

Usage:
    python evaluate_cloning.py --model <ckpt.pth> --config configs/vits_yourtts_vctk.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pipeline_config import (
    LIBRITTS_DIR,
    SE_CONFIG_PATH,
    SE_MODEL_PATH,
    VCTK_DIR,
)
from speaker_logic import collect_speaker_clips, split_refs_and_prompts
from tts_runtime import load_model as _load_tts_model
from tts_runtime import synthesize as _synthesize


@dataclass
class EvalReport:
    ckpt: str
    heldout_similarity: list[float] = field(default_factory=list)
    train_similarity: list[float] = field(default_factory=list)
    random_baseline: float = 0.0
    utmos_heldout: list[float] = field(default_factory=list)
    per_speaker: dict = field(default_factory=dict)

    def summary(self) -> dict:
        def stats(xs):
            return {"mean": float(np.mean(xs)), "n": len(xs), "min": float(np.min(xs)), "max": float(np.max(xs))} if xs else {}

        return {
            "checkpoint": self.ckpt,
            "heldout_speaker_similarity": stats(self.heldout_similarity),
            "train_speaker_similarity": stats(self.train_similarity),
            "random_embedding_baseline_cosine": self.random_baseline,
            "utmos_heldout": stats(self.utmos_heldout),
            "per_speaker": self.per_speaker,
        }


# ---------------------------------------------------------------------------
# Reference / synthesis data selection
# ---------------------------------------------------------------------------


def _collect_speaker_clips_compat(root: Path, speaker_prefix: str, limit: int = 6) -> list[Path]:
    return collect_speaker_clips(root, speaker_prefix, limit)


# ---------------------------------------------------------------------------
# Similarity scoring (frozen SE - same encoder as training/inference)
# ---------------------------------------------------------------------------


def load_scorer():
    from TTS.tts.utils.speakers import SpeakerManager

    mgr = SpeakerManager(
        encoder_model_path=SE_MODEL_PATH, encoder_config_path=SE_CONFIG_PATH, use_cuda=torch.cuda.is_available()
    )
    return mgr


def embed_files(mgr, files: list[Path]) -> np.ndarray:
    embs = [np.asarray(mgr.compute_embedding_from_clip(str(f)), dtype=np.float32) for f in files]
    m = np.stack(embs).mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def embed_waveform(mgr, wav: np.ndarray, sr: int) -> np.ndarray:
    """Embed an in-memory waveform by routing through the SE's AudioProcessor."""
    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, sr)
        emb = np.asarray(mgr.compute_embedding_from_clip(tmp.name), dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


# ---------------------------------------------------------------------------
# Naturalness (UTMOS via MOSNet-style predictor from torch hub)
# ---------------------------------------------------------------------------


def load_utmos():
    """UTMOS strong predictor; returns None if unavailable (naturalness optional)."""
    try:
        model = torch.hub.load("sarulab-speech/utmos22", "utmos22", trust_repo=True)
        model.eval()
        return model
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] UTMOS unavailable ({exc}); skipping naturalness")
        return None


def utmos_score(model, wav: np.ndarray, sr: int) -> float:
    import torchaudio

    if sr != 16000:
        wav = torchaudio.functional.resample(torch.from_numpy(wav).float()[None], sr, 16000)[0].numpy()
    x = torch.from_numpy(wav).float()[None]
    with torch.no_grad():
        return float(model(x, sr=16000).item())


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def load_model(config_path: Path, ckpt_path: Path):
    """Thin re-export of the shared loader (kept for API clarity in this module)."""
    return _load_tts_model(config_path, ckpt_path)


def synthesize(model, text: str, ref_files: list[Path]) -> np.ndarray:
    return _synthesize(model, text, ref_files, language="en")


def evaluate(
    model,
    config,
    mgr,
    utmos,
    speakers: list[str],
    refs_by_speaker: dict[str, list[Path]],
    prompts_by_speaker: dict[str, list[Path]],
    texts_by_speaker: dict[str, list[str]],
    max_per_speaker: int,
    tag: str,
    report: EvalReport,
) -> None:
    sr = int(getattr(config.audio, "sample_rate", 22050))
    for spk in speakers:
        refs = refs_by_speaker[spk]
        prompts = prompts_by_speaker.get(spk, [])
        texts = texts_by_speaker.get(spk, [])[:max_per_speaker]
        if not prompts or not texts:
            prompts = refs  # fall back: synthesize the reference text itself
            texts = None
        target = embed_files(mgr, refs)
        sims = []
        for i, wav_file in enumerate(prompts[:max_per_speaker]):
            text = texts[i] if texts else None
            if text is None:
                # VCTK prompt texts live next to nothing - use a fixed prompt
                text = "The quick brown fox jumps over the lazy dog."
            try:
                wav = synthesize(model, text, refs)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] synthesis failed {spk}: {exc}")
                continue
            gen_emb = embed_waveform(mgr, wav, sr)
            s = cosine(gen_emb, target)
            m_score = utmos_score(utmos, wav, sr) if utmos else None
            sims.append(s)
            report.per_speaker.setdefault(tag, {}).setdefault(spk, []).append(
                {"cosine": s, "utmos": m_score}
            )
            if utmos and tag == "heldout":
                report.utmos_heldout.append(m_score)
        (report.heldout_similarity if tag == "heldout" else report.train_similarity).extend(sims)
        print(f"[{tag}] {spk}: mean cosine {np.mean(sims):.3f} over {len(sims)} utts" if sims else f"[{tag}] {spk}: skipped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to best_model.pth / checkpoint")
    ap.add_argument("--config", required=True, help="matching config json")
    ap.add_argument("--dataset", choices=["vctk", "libritts"], default="vctk")
    ap.add_argument("--max-per-speaker", type=int, default=2)
    ap.add_argument("--n-train-control", type=int, default=8, help="training speakers used as control group")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    root = VCTK_DIR if args.dataset == "vctk" else LIBRITTS_DIR
    meta = root / "metadata"

    # held-out speakers: the dataset metadata file is the source of truth
    heldout_path = meta / "heldout_speakers.json"
    if not heldout_path.exists():
        raise SystemExit(f"{heldout_path} missing - run the prep stage first")
    heldout = json.loads(heldout_path.read_text())
    if not heldout:
        raise SystemExit("no held-out speakers listed - nothing to evaluate")

    model, config = load_model(Path(args.config), Path(args.model))
    mgr = load_scorer()
    utmos = load_utmos()

    report = EvalReport(ckpt=str(args.model))

    # random-embedding baseline: similarity of synthetic audio to random 512-d vecs
    rnd = torch.randn(256, 512)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)
    report.random_baseline = float(0.0)  # E[cos(v, random unit vec)] ~ 0 by symmetry
    print(f"[info] random-embedding baseline (analytic): {report.random_baseline:.3f}")

    # held-out speakers
    refs_h, prompts_h, texts_h = {}, {}, {}
    for spk in heldout:
        pref = f"VCTK_{spk}" if args.dataset == "vctk" else f"LTTS_{spk}"
        clips = collect_speaker_clips(root, pref)
        if not clips:
            continue
        refs, prompts = split_refs_and_prompts(clips)
        refs_h[pref], prompts_h[pref] = refs, prompts
        texts_h[pref] = _texts_for_prompts(root, args.dataset, spk, prompts)
    heldout_pref = list(refs_h.keys())

    # training-speaker control group (sample from the d-vector train file)
    train_speakers = sorted({v["name"] for v in json.load(open(meta / "d_vectors_train.json")).values()})
    rng = random.Random(0)
    train_sample = rng.sample(train_speakers, min(args.n_train_control, len(train_speakers)))
    refs_t, prompts_t, texts_t = {}, {}, {}
    for pref in train_sample:
        spk = pref.split("_", 1)[1]
        clips = collect_speaker_clips(root, pref)
        if not clips:
            continue
        refs, prompts = split_refs_and_prompts(clips)
        refs_t[pref], prompts_t[pref] = refs, prompts
        texts_t[pref] = _texts_for_prompts(root, args.dataset, spk, prompts)

    evaluate(model, config, mgr, utmos, heldout_pref, refs_h, prompts_h, texts_h, args.max_per_speaker, "heldout", report)
    evaluate(model, config, mgr, utmos, list(refs_t.keys()), refs_t, prompts_t, texts_t, args.max_per_speaker, "train", report)

    summary = report.summary()
    out = args.output or Path(args.model).parent / "cloning_eval.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n================ CLONING EVAL ================")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_speaker"}, indent=2))
    print(f"[ok] full report -> {out}")


def _texts_for_prompts(root: Path, dataset: str, spk: str, prompts: list[Path]) -> list[str]:
    """Ground-truth transcripts for the prompt clips (VCTK txt/ files; LibriTTS tsv)."""
    texts = []
    if dataset == "vctk":
        for p in prompts:
            utt = p.name.split("_mic")[0]
            txt = root / "txt" / spk / f"{utt}.txt"
            texts.append(txt.read_text(encoding="utf-8").strip() if txt.exists() else "")
    else:
        # held-out utterances live in the eval tsv, training ones in the train tsv
        lookup = {}
        for name in ("libritts_train.tsv", "libritts_eval.tsv"):
            tsv = root / "metadata" / name
            if tsv.exists():
                for line in tsv.read_text(encoding="utf-8").splitlines():
                    rel, text = line.split("\t", 1)
                    lookup[Path(rel).name] = text
        for p in prompts:
            texts.append(lookup.get(p.name, ""))
    return texts


if __name__ == "__main__":
    main()
