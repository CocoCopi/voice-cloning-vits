"""Pure (torch-free) helpers for speaker identity, d-vector keys and eval splits.

Kept import-light on purpose: the d-vector key convention is the single most
fragile contract in the pipeline, so it lives in one testable place.
"""

from __future__ import annotations

from pathlib import Path


def raw_speaker_from_path(wav: Path | str) -> str:
    """Bare speaker id as it appears in heldout_speakers.json ('p225', '84').

    Parse the UTTERANCE STEM first: both corpora encode the speaker as the
    first underscore-delimited token of the filename
      - VCTK:    p225_001_mic1.flac  -> 'p225'
      - LibriTTS: 84_121550_000007_000000.wav -> '84'
    Falling back to directory scanning alone is WRONG for LibriTTS because
    <chapter>/ precedes <speaker>/ in the path.
    """
    first = Path(wav).stem.split("_")[0]
    if first.isdigit() or (first.startswith("p") and first[1:].isdigit()):
        return first
    for p in reversed(Path(wav).parts):
        if (p.startswith("p") and p[1:].isdigit()) or p.isdigit():
            return p
    raise ValueError(f"cannot infer speaker from {wav}")


def speaker_from_path(wav: Path | str) -> str:
    """Prefixed speaker name used as the embedding 'name' (matches formatters)."""
    raw = raw_speaker_from_path(wav)
    return f"VCTK_{raw}" if raw.startswith("p") else f"LTTS_{raw}"


def rel_key(wav: Path | str, root: Path | str, dataset_name: str) -> str:
    """d-vector JSON key. MUST equal Coqui's `audio_unique_name` format:
    f"{dataset_name}#{relpath-without-extension}" (see TTS add_extra_keys).
    """
    rel = Path(wav).resolve().relative_to(Path(root).resolve()).with_suffix("")
    return f"{dataset_name}#{rel}"


def collect_speaker_clips(root: Path, speaker_prefix: str, limit: int = 6) -> list[Path]:
    """Reference-clip candidates for a speaker ('VCTK_p225' or 'LTTS_84')."""
    spk = speaker_prefix.split("_", 1)[1]
    root = Path(root)
    if speaker_prefix.startswith("VCTK_"):
        d = root / "wav48_silence_trimmed" / spk
        return (sorted(d.glob("*_mic1.flac")) if d.is_dir() else [])[:limit]
    return sorted(root.rglob(f"{spk}_*.wav"))[:limit]


def split_refs_and_prompts(clips: list[Path], n_refs: int = 2) -> tuple[list[Path], list[Path]]:
    """First n clips = references; the rest = synthesis prompts (disjoint utts)."""
    if len(clips) < n_refs + 1:
        n_refs = max(1, len(clips) - 1)
    return clips[:n_refs], clips[n_refs:]
