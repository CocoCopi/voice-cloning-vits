"""Zero-shot voice cloning inference: ANY reference clip + text -> cloned speech.

Works for speakers never seen in training because conditioning is a d-vector
computed by the frozen pretrained speaker encoder - exactly the embedding space
the TTS decoder learned to speak from.

Usage:
    python infer.py --model output/vits_vctk/best_model.pth \
        --config configs/vits_yourtts_vctk.json \
        --ref my_voice_sample.wav --text "Hello, this is a cloned voice."
    python infer.py ... --refs ref1.wav ref2.wav ref3.wav   # better: average refs
    python infer.py ... --text-file input.txt               # whole file
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf

from tts_runtime import load_model, synthesize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref", action="append", default=[], help="reference clip(s); repeat for averaging")
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    refs = [Path(p) for p in args.ref]
    if not refs:
        raise SystemExit("provide at least one --ref clip (3-10 s clean speech works best)")
    for r in refs:
        if not r.exists():
            raise SystemExit(f"reference not found: {r}")
    text = args.text or (args.text_file.read_text(encoding="utf-8").strip() if args.text_file else None)
    if not text:
        raise SystemExit("provide --text or --text-file")

    model, config = load_model(args.config, args.model)
    sr = config.audio.sample_rate

    t0 = time.time()
    wav = synthesize(model, text, refs, language=args.language)
    dt = time.time() - t0

    out = args.out or Path("cloned_output.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wav, sr)
    dur = len(wav) / sr
    print(f"[ok] {out} | {dur:.1f}s audio in {dt:.1f}s (RTF {dt / max(dur, 1e-6):.2f})")


if __name__ == "__main__":
    main()
