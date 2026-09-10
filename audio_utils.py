"""Audio utilities: in-place resampling to the TTS training sample rate.

`VitsDataset.load_audio` (Coqui) loads files at their NATIVE sample rate and
never resamples, while the mel filterbank is built from `config.audio.sample_rate`.
Feeding non-22050 Hz audio trains a technically-converging model with wrong
frequency mapping. So we normalize every corpus to 22050 Hz mono ONCE here.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

TARGET_SR = 22_050


def _resample_one(args: tuple[str, int]) -> str:
    path, target_sr = args
    import soundfile as sf

    try:
        info = sf.info(path)
        if info.samplerate == target_sr and info.channels == 1:
            return "skip"
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)  # mono
        if sr != target_sr:
            try:
                import torch
                import torchaudio

                wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
                data = wav.squeeze(0).numpy()
            except ImportError:
                # torchaudio-free fallback (linear interpolation; adequate for
                # corpus normalization, Colab uses the polyphase path above)
                import numpy as np

                n_out = int(round(len(data) * target_sr / sr))
                data = np.interp(
                    np.linspace(0.0, len(data) - 1, num=n_out, dtype=np.float64),
                    np.arange(len(data), dtype=np.float64),
                    data.astype(np.float64),
                ).astype(np.float32)
        sf.write(path, data, target_sr, format="FLAC" if path.lower().endswith(".flac") else "WAV")
        return "ok"
    except Exception as exc:  # noqa: BLE001 - keep going on bad files
        return f"error: {exc}"


def resample_tree(
    files: list[Path] | Path,
    target_sr: int = TARGET_SR,
    workers: int = 8,
    desc: str = "resample",
) -> dict[str, int]:
    """Resample/mono-ize audio files in place. `files` may be a directory (rglob flac+wav)."""
    if isinstance(files, Path):
        files = sorted(p for p in files.rglob("*") if p.suffix.lower() in {".flac", ".wav"})
    paths = [str(p) for p in files]
    stats = {"ok": 0, "skip": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for status in tqdm(
            ex.map(_resample_one, [(p, target_sr) for p in paths], chunksize=64),
            total=len(paths),
            desc=desc,
            unit="file",
        ):
            if status.startswith("error"):
                stats["error"] += 1
                tqdm.write(status)
            elif status == "skip":
                stats["skip"] += 1
            else:
                stats["ok"] += 1
    return stats
