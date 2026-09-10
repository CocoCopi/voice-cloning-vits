"""Fetch VCTK 0.92 and lay it out for Coqui's stock `vctk` formatter.

Target layout (what TTS.tts.formatters.vctk expects):
    vctk/
      txt/<speaker>/<utt>.txt              # one-line transcripts
      wav48_silence_trimmed/<speaker>/<utt>_mic1.flac   (mic2 for most speakers)

Primary path: HuggingFace `datasets` parquet of `confit/vctk-full` (audio +
transcript rows, ~11 GB) -> written to disk in the layout above. This avoids
the slow single-file Edinburgh zip when a Colab-friendly mirror is faster.
Fallback: direct Edinburgh DataShare zip (also ~11 GB) with optional Drive copy.

Usage:
    python prepare_vctk.py                    # HF parquet -> files on disk
    python prepare_vctk.py --from-zip PATH    # skip download, use local zip
    python prepare_vctk.py --zip-only         # force Edinburgh zip route
    python prepare_vctk.py --verify           # re-run layout validation
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

from audio_utils import resample_tree
from pipeline_config import VCTK_DIR

HF_DATASET = "confit/vctk-full"
EDINBURGH_ZIP = "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"
SPEAKER_RE = re.compile(r"(p\d{3})")
BAD_SPEAKERS = {"p315", "p280"}  # known corrupted mic2/mic1 sessions; keep mic1 copies anyway
EXPECTED_MIN_SPEAKERS = 105
EXPECTED_MIN_UTTS = 40_000


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_layout(root: Path, quiet: bool = False) -> bool:
    txt_root, wav_root = root / "txt", root / "wav48_silence_trimmed"
    if not txt_root.is_dir() or not wav_root.is_dir():
        return False
    speakers = sorted(p.name for p in wav_root.iterdir() if p.is_dir())
    n_utts = sum(1 for _ in wav_root.rglob("*.flac"))
    ok = len(speakers) >= EXPECTED_MIN_SPEAKERS and n_utts >= EXPECTED_MIN_UTTS
    if not quiet or ok:
        print(f"[{'ok' if ok else 'warn'}] VCTK layout: {len(speakers)} speakers, {n_utts} flac files")
    return ok


# ---------------------------------------------------------------------------
# Route A - HuggingFace parquet
# ---------------------------------------------------------------------------


def prepare_from_hf(root: Path) -> None:
    import pyarrow.parquet as pq
    from datasets import load_dataset

    if validate_layout(root, quiet=True):
        print("[ok] VCTK already prepared - skipping HF download")
        return

    print(f"[hf] streaming {HF_DATASET} -> {root}")
    ds = load_dataset(HF_DATASET, split="train", streaming=False)
    txt_root = root / "txt"
    wav_root = root / "wav48_silence_trimmed"
    txt_root.mkdir(parents=True, exist_ok=True)
    wav_root.mkdir(parents=True, exist_ok=True)

    n_written = 0
    id2spk: dict[str, str] = {}
    for row in tqdm(ds, desc="writing VCTK", unit="utt", total=len(ds)):
        audio = row["audio"]
        path = audio.get("path") or ""
        m = SPEAKER_RE.search(Path(path).name)
        if not m:
            continue
        spk = m.group(1)
        utt_id = Path(path).stem  # e.g. p225_001
        id2spk[utt_id] = spk
        spk_txt_dir = txt_root / spk
        spk_wav_dir = wav_root / spk
        spk_txt_dir.mkdir(parents=True, exist_ok=True)
        spk_wav_dir.mkdir(parents=True, exist_ok=True)
        # transcript
        text = (row.get("text") or "").strip()
        if text and not (spk_txt_dir / f"{utt_id}.txt").exists():
            (spk_txt_dir / f"{utt_id}.txt").write_text(text + "\n", encoding="utf-8")
        # audio (flac)
        out_flac = spk_wav_dir / f"{utt_id}_mic1.flac"
        if not out_flac.exists():
            sf_write_flac(audio["array"], audio["sampling_rate"], out_flac)
        n_written += 1
    print(f"[ok] wrote {n_written} utterances")


_SF_CACHE: dict[str, object] = {}


def sf_write_flac(array, sr: int, out: Path) -> None:
    import soundfile as sf

    if array.ndim == 2:  # (T, C) -> mono
        array = array.mean(axis=1)
    sf.write(out, array, sr, format="FLAC")


# ---------------------------------------------------------------------------
# Route B - Edinburgh zip (single ~11 GB file)
# ---------------------------------------------------------------------------


def prepare_from_zip(zip_path: Path, root: Path) -> None:
    print(f"[zip] extracting {zip_path} -> {root}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        wanted = [m for m in members if "/txt/" in m or "wav48_silence_trimmed" in m]
        for m in tqdm(wanted, desc="unzip", unit="file"):
            zf.extract(m, root)
    validate_layout(root)


def download_edinburgh_zip(dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dl] Edinburgh VCTK zip -> {dest}")
    with requests.get(EDINBURGH_ZIP, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="vctk.zip") as pbar:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-zip", type=Path, default=None, help="extract an existing VCTK 0.92 zip")
    ap.add_argument("--zip-only", action="store_true", help="force the Edinburgh zip download route")
    ap.add_argument("--zip-dest", type=Path, default=None, help="where to keep the zip (default: DATA_ROOT)")
    ap.add_argument("--verify", action="store_true", help="only validate existing layout")
    args = ap.parse_args()

    root = VCTK_DIR
    if args.verify:
        sys.exit(0 if validate_layout(root) else 1)

    if args.from_zip:
        prepare_from_zip(args.from_zip, root)
    elif args.zip_only:
        zp = args.zip_dest or (root.parent / "DS_10283_3443.zip")
        if not zp.exists():
            download_edinburgh_zip(zp)
        prepare_from_zip(zp, root)
    else:
        try:
            prepare_from_hf(root)
            if not validate_layout(root, quiet=True):
                raise RuntimeError("HF route produced incomplete layout")
        except Exception as exc:
            print(f"[warn] HF route failed ({exc}); falling back to Edinburgh zip")
            zp = args.zip_dest or (root.parent / "DS_10283_3443.zip")
            if not zp.exists():
                download_edinburgh_zip(zp)
            prepare_from_zip(zp, root)

    if not validate_layout(root):
        print("[error] VCTK validation failed")
        sys.exit(1)
    # Coqui's VitsDataset loads audio at native SR and never resamples, while the
    # mel filterbank is built for config.audio.sample_rate (22050). Normalize once:
    print("[info] normalizing VCTK to 22050 Hz mono ...")
    print(resample_tree(root / "wav48_silence_trimmed"))
    print("[done] VCTK ready")


if __name__ == "__main__":
    main()
