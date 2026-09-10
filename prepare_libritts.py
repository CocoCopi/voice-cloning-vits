"""Fetch LibriTTS, build a speaker-diverse subset for VITS training.

Speakers are the generalization lever: a 10-utt/speaker cap across ~2,456
speakers gives ~17k utterances of maximum speaker diversity in a fraction of
the training time - exactly the YourTTS recipe rationale.

Pipeline:
  1. download train-clean-100 (+ train-clean-360 with --with-360) from OpenSLR
  2. extract, validate layout
  3. cap utterances per speaker, cap speaker count (optional)
  4. hold out ~5% of test speakers (seeded, never trained on)
  5. emit train/eval metadata TSVs + heldout_speakers.json

Usage:
    python prepare_libritts.py                       # train-clean-100, 10 utt/spk
    python prepare_libritts.py --max-utts-per-spk 6  # leaner subset
    python prepare_libritts.py --with-360            # add train-clean-360
    python prepare_libritts.py --skip-download       # reuse existing tarballs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

from audio_utils import resample_tree
from pipeline_config import HOLDOUT_SEED, LIBRITTS_DIR

OPENSLR = {
    "train-clean-100": (
        "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
        "2a6abacc1c854b76e97fa4dc4d99398c",  # md5
    ),
    "train-clean-360": (
        "https://www.openslr.org/resources/60/train-clean-360.tar.gz",
        "ff8bb2a850aab049cf1f9016e3a23e5a",  # md5
    ),
}
MIN_UTTS_PER_SPK = 3  # fewer than this and the speaker adds little to cloning
HOLDOUT_FRACTION = 0.05  # ~5% of subset speakers reserved as unseen test voices


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_md5 is None or md5sum(dest) == expected_md5:
            print(f"[ok] cached {dest.name}")
            return dest
        print(f"[warn] {dest.name} failed md5 - redownloading")
        dest.unlink()
    print(f"[dl] {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                pbar.update(len(chunk))
    if expected_md5 and (got := md5sum(dest)) != expected_md5:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"md5 mismatch for {dest.name}: got {got}, want {expected_md5}")
    return dest


def extract(tarball: Path, dest_root: Path) -> None:
    with tarfile.open(tarball, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc=f"extract {tarball.name}", unit="file"):
            tf.extract(member, dest_root)


def validate_layout(root: Path) -> int:
    wavs = list(root.rglob("*.wav"))
    spks = {p.name for p in root.rglob("*") if p.is_dir() and p.name.isdigit()}
    print(f"[info] LibriTTS layout: {len(spks)} speaker dirs, {len(wavs)} wavs")
    return len(wavs)


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def build_subset(
    root: Path,
    max_utts_per_spk: int,
    max_speakers: int,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """Return (train_rows, eval_rows, heldout_ids).

    Rows are (relpath, normalized text) pairs, relative to `root` so metadata
    survives directory moves. A speaker is either entirely in-train or entirely
    held out - never both - and the split is seeded so every session sees the
    same test speakers.
    """
    utts_by_spk: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for wav in sorted(root.rglob("*.wav")):
        spk = wav.name.split("_")[0]
        trans = wav.with_name(wav.stem + ".normalized.txt")
        if not trans.exists():
            trans = wav.with_name(wav.stem + ".original.txt")
        if not trans.exists():
            continue
        text = trans.read_text(encoding="utf-8").strip()
        if not text or any(ch.isdigit() for ch in text):
            continue  # skip digit sentences; normalized text spells numbers out
        utts_by_spk[spk].append((wav, text))

    # speakers with enough usable utterances
    eligible = sorted(sp for sp, rows in utts_by_spk.items() if len(rows) >= MIN_UTTS_PER_SPK)

    # deterministic holdout: after optional speaker sampling, the first
    # HOLDOUT_FRACTION of sorted speaker ids become unseen test voices
    rng = random.Random(seed)
    if max_speakers and len(eligible) > max_speakers:
        eligible = sorted(rng.sample(eligible, max_speakers))

    train_rows: list[tuple[str, str]] = []
    eval_rows: list[tuple[str, str]] = []
    heldout: list[str] = []
    for i, spk in enumerate(eligible):
        rows = sorted(utts_by_spk[spk])
        take = min(max_utts_per_spk, len(rows))
        rel_rows = [(str(w.relative_to(root)), t) for w, t in rows[:take]]
        if i < int(HOLDOUT_FRACTION * len(eligible)):
            heldout.append(spk)
            eval_rows.extend(rel_rows)
        else:
            train_rows.extend(rel_rows)
    return train_rows, eval_rows, heldout


def write_metadata(rows: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for wav, text in rows:
            f.write(f"{wav}\t{text}\n")
    print(f"[ok] {path.name}: {len(rows)} rows -> {path}")


def ensure_metadata(
    root: Path, max_utts_per_spk: int, max_speakers: int, seed: int
) -> tuple[Path, Path, list[str]]:
    """Guarantee train/eval metadata exists; regenerate on demand.

    Lets the notebook/training entrypoint call this even when the prep stage
    ran in a previous session and only metadata files went missing.
    """
    meta_dir = root / "metadata"
    train_p, eval_p = meta_dir / "libritts_train.tsv", meta_dir / "libritts_eval.tsv"
    if train_p.exists() and eval_p.exists():
        hp = meta_dir / "heldout_speakers.json"
        heldout = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else []
        return train_p, eval_p, heldout
    train_rows, eval_rows, heldout = build_subset(root, max_utts_per_spk, max_speakers, seed)
    write_metadata(train_rows, train_p)
    write_metadata(eval_rows, eval_p)
    (meta_dir / "heldout_speakers.json").write_text(json.dumps(heldout, indent=2), encoding="utf-8")
    return train_p, eval_p, heldout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts-per-spk", type=int, default=10)
    ap.add_argument("--max-speakers", type=int, default=0, help="0 = no cap")
    ap.add_argument("--with-360", action="store_true", help="also download train-clean-360")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    args = ap.parse_args()

    root = LIBRITTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    subsets = ["train-clean-100"] + (["train-clean-360"] if args.with_360 else [])

    if not args.skip_download:
        for name in subsets:
            url, md5 = OPENSLR[name]
            tarball = root / f"{name}.tar.gz"
            download(url, tarball, md5)
            marker = root / f".{name}_extracted"
            if not marker.exists():
                extract(tarball, root)
                marker.touch()

    total = validate_layout(root)
    if total < 1000:
        print("[error] LibriTTS layout looks wrong - check extraction")
        raise SystemExit(1)

    # Coqui's VitsDataset loads audio at native SR and never resamples, while the
    # mel filterbank is built for config.audio.sample_rate (22050). Normalize once:
    print("[info] normalizing LibriTTS to 22050 Hz mono ...")
    print(resample_tree(root, desc="resample libritts"))

    train_rows, eval_rows, heldout = build_subset(root, args.max_utts_per_spk, args.max_speakers, args.seed)
    meta_dir = root / "metadata"
    write_metadata(train_rows, meta_dir / "libritts_train.tsv")
    write_metadata(eval_rows, meta_dir / "libritts_eval.tsv")
    (meta_dir / "heldout_speakers.json").write_text(json.dumps(heldout, indent=2), encoding="utf-8")
    print(f"[done] train={len(train_rows)} eval={len(eval_rows)} heldout_speakers={len(heldout)}")
    print(f"[info] heldout: {heldout[:10]}{'...' if len(heldout) > 10 else ''}")


if __name__ == "__main__":
    main()
