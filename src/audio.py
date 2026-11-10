"""Audio helpers: noise injection and resampling."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from src.config import RANDOM_SEED, TARGET_SAMPLE_RATE


def add_noise(audio: np.ndarray, snr_db: float, key: str = "") -> np.ndarray:
    """
    Add white Gaussian noise so that the result has the requested
    signal-to-noise ratio (in dB).

        SNR_dB = 10 * log10(P_signal / P_noise)   =>   P_noise = P_signal / 10^(SNR/10)

    `key` (e.g. the sample id) seeds the noise, so the same sample gets the
    same noise for every model: the comparison stays fair and re-runs are
    reproducible.
    """
    audio = np.asarray(audio, dtype=np.float32)
    power = float(np.mean(audio ** 2))
    if power == 0.0:
        return audio
    noise_power = power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(RANDOM_SEED + zlib.crc32(key.encode()))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return audio + noise


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    return float(10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2)))


def mix_background_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix mono noise to an exact target SNR without clipping or normalization."""
    clean = np.asarray(audio, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if clean.ndim != 1 or noise.ndim != 1:
        raise ValueError("background-noise mixing expects mono waveforms")
    if clean.size == 0 or noise.size == 0:
        raise ValueError("audio and background noise must be non-empty")
    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise ** 2))
    if signal_power <= 0 or noise_power <= 0:
        raise ValueError("audio and background noise must have non-zero power")
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]
    # Recompute after crop/repeat; the selected segment can have a different RMS.
    noise_power = float(np.mean(noise ** 2))
    scaled_noise = noise * np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return clean + scaled_noise


def add_background_noise(audio: np.ndarray, snr_db: float, key: str, noise_dir: str) -> np.ndarray:
    """Select a repeatable WAV from a user-provided noise directory and mix it."""
    import soundfile as sf

    files = sorted(Path(noise_dir).glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav background-noise recordings found in {noise_dir}")
    seed = RANDOM_SEED + zlib.crc32(key.encode())
    path = files[seed % len(files)]
    noise, sr = sf.read(path, dtype="float32", always_2d=True)
    noise = noise.mean(axis=1)
    noise, _ = to_mono_16k(noise, sr)
    if len(noise) > len(audio):
        start = (seed // max(1, len(files))) % (len(noise) - len(audio) + 1)
        noise = noise[start:start + len(audio)]
    return mix_background_noise(audio, noise, snr_db)


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Downmix to mono float32 and resample to 16 kHz if needed."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1) if audio.shape[1] <= 8 else audio.mean(axis=0)
    if sample_rate != TARGET_SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE
    return audio, sample_rate
