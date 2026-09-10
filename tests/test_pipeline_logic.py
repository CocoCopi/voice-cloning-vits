"""Pure-logic tests for the fragile contracts in the pipeline (no torch/TTS needed).

The d-vector key convention is the single most failure-prone piece of the
whole system: if it drifts from Coqui's `audio_unique_name` format, training
crashes with a KeyError on the first batch. These tests pin it down.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from audio_utils import TARGET_SR, _resample_one
from speaker_logic import (
    collect_speaker_clips,
    raw_speaker_from_path,
    rel_key,
    speaker_from_path,
    split_refs_and_prompts,
)

# ---------------------------------------------------------------------------
# d-vector key convention (Coqui add_extra_keys compatibility)
# ---------------------------------------------------------------------------


def test_rel_key_vctk_matches_coqui_audio_unique_name(tmp_path):
    # Coqui vctk formatter: speaker_name="VCTK_p225", root=VCTK_DIR, file
    # wav48_silence_trimmed/p225/p225_001_mic1.flac
    # add_extra_keys -> f"{dataset}#{relpath-without-ext}"
    wav = tmp_path / "wav48_silence_trimmed" / "p225" / "p225_001_mic1.flac"
    assert rel_key(wav, tmp_path, "vctk") == "vctk#wav48_silence_trimmed/p225/p225_001_mic1"


def test_rel_key_libritts_matches_coqui_audio_unique_name(tmp_path):
    # libritts_subset formatter keeps LibriTTS layout <spk>/<chapter>/<utt>.wav
    wav = tmp_path / "84" / "121550" / "84_121550_000007_000000.wav"
    assert rel_key(wav, tmp_path, "libritts") == "libritts#84/121550/84_121550_000007_000000"


# ---------------------------------------------------------------------------
# Speaker identity parsing (must match Coqui formatter prefixes)
# ---------------------------------------------------------------------------


def test_speaker_from_path_vctk():
    p = Path("/x/wav48_silence_trimmed/p225/p225_001_mic1.flac")
    assert raw_speaker_from_path(p) == "p225"
    assert speaker_from_path(p) == "VCTK_p225"


def test_speaker_from_path_libritts():
    p = Path("/x/LibriTTS/84/121550/84_121550_000007_000000.wav")
    assert raw_speaker_from_path(p) == "84"
    assert speaker_from_path(p) == "LTTS_84"


def test_speaker_from_path_rejects_garbage():
    with pytest.raises(ValueError):
        raw_speaker_from_path(Path("/some/weird/file.flac"))


# ---------------------------------------------------------------------------
# Eval clip splitting (references must be disjoint from prompts)
# ---------------------------------------------------------------------------


def test_split_refs_and_prompts_disjoint():
    clips = [Path(f"c{i}.wav") for i in range(6)]
    refs, prompts = split_refs_and_prompts(clips, n_refs=2)
    assert refs == clips[:2]
    assert prompts == clips[2:]
    assert not set(refs) & set(prompts)


def test_split_refs_and_prompts_tiny_speaker():
    clips = [Path("a.wav"), Path("b.wav")]
    refs, prompts = split_refs_and_prompts(clips, n_refs=2)
    assert refs == [Path("a.wav")]  # forced to 1 ref
    assert prompts == [Path("b.wav")]


def test_collect_speaker_clips_vctk_uses_mic1(tmp_path):
    d = tmp_path / "wav48_silence_trimmed" / "p225"
    d.mkdir(parents=True)
    (d / "p225_001_mic1.flac").touch()
    (d / "p225_002_mic2.flac").touch()
    clips = collect_speaker_clips(tmp_path, "VCTK_p225")
    assert clips == [d / "p225_001_mic1.flac"]


# ---------------------------------------------------------------------------
# Resample helper (skip/error contract used by both prep scripts)
# ---------------------------------------------------------------------------


def test_resample_skip_on_correct_sr(tmp_path):
    sf = pytest.importorskip("soundfile")
    p = tmp_path / "x.wav"
    sf.write(p, [0.0, 0.1, -0.1], TARGET_SR, subtype="FLOAT")
    status = _resample_one((str(p), TARGET_SR))
    assert status == "skip"


def test_resample_downsamples(tmp_path):
    sf = pytest.importorskip("soundfile")
    p = tmp_path / "y.wav"
    sf.write(p, [0.0] * 8000, 44100, subtype="FLOAT")
    status = _resample_one((str(p), TARGET_SR))
    assert status == "ok"
    info = sf.info(p)
    assert info.samplerate == TARGET_SR


# ---------------------------------------------------------------------------
# LibriTTS subset builder (seeded holdout, speaker integrity)
# ---------------------------------------------------------------------------


def _make_libritts_tree(root: Path, speakers: int = 20, utts: int = 5) -> None:
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dogs"]
    for s in range(speakers):
        spk = 100 + s
        for u in range(utts):
            d = root / str(spk) / str(121550 + u)
            d.mkdir(parents=True, exist_ok=True)
            wav = d / f"{spk}_{121550 + u}_00000{u}_000000.wav"
            wav.touch()
            # digit-free text: the subset builder skips sentences with digits
            text = " ".join(words[(s + u + k) % len(words)] for k in range(4))
            (d / f"{spk}_{121550 + u}_00000{u}_000000.normalized.txt").write_text(text, encoding="utf-8")


def test_build_subset_holdout_integrity(tmp_path):
    import prepare_libritts as pl

    _make_libritts_tree(tmp_path)
    train, eval_rows, heldout = pl.build_subset(tmp_path, max_utts_per_spk=3, max_speakers=0, seed=7)
    assert heldout, "expected a nonempty held-out split"
    assert len(train) > len(eval_rows)
    # a speaker is entirely in-train or entirely held out
    train_spks = {Path(w).parts[0] for w, _ in train}
    eval_spks = {Path(w).parts[0] for w, _ in eval_rows}
    assert not (train_spks & eval_spks), "speaker leaked into both splits"
    assert eval_spks == set(heldout)
    # deterministic across calls
    train2, eval2, heldout2 = pl.build_subset(tmp_path, max_utts_per_spk=3, max_speakers=0, seed=7)
    assert (train, eval_rows, heldout) == (train2, eval2, heldout2)


def test_build_subset_caps_utts_per_speaker(tmp_path):
    import prepare_libritts as pl

    _make_libritts_tree(tmp_path, speakers=10, utts=8)
    train, _, _ = pl.build_subset(tmp_path, max_utts_per_spk=4, max_speakers=0, seed=1)
    per_spk: dict[str, int] = {}
    for w, _ in train:
        per_spk[Path(w).parts[0]] = per_spk.get(Path(w).parts[0], 0) + 1
    assert all(v <= 4 for v in per_spk.values())


def test_build_subset_skips_digit_sentences(tmp_path):
    import prepare_libritts as pl

    spk = 300
    d = tmp_path / str(spk) / "121550"
    d.mkdir(parents=True)
    # sentence with digits (unusable) vs a clean one
    (d / f"{spk}_121550_000000_000000.wav").touch()
    (d / f"{spk}_121550_000000_000000.normalized.txt").write_text("call 555 0198 now", encoding="utf-8")
    (d / f"{spk}_121550_000001_000000.wav").touch()
    (d / f"{spk}_121550_000001_000000.normalized.txt").write_text("clean sentence here", encoding="utf-8")
    train, _, _ = pl.build_subset(tmp_path, max_utts_per_spk=5, max_speakers=0, seed=1)
    assert all("555" not in t for _, t in train)
