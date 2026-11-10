import numpy as np
import pytest

from src.audio import add_noise, measured_snr_db, mix_background_noise, to_mono_16k


@pytest.mark.parametrize("snr", [20.0, 10.0, 5.0])
def test_noise_hits_requested_snr(snr):
    t = np.linspace(0, 2, 32000, dtype=np.float32)
    clean = 0.5 * np.sin(2 * np.pi * 220 * t)
    noisy = add_noise(clean, snr, key="s1")
    assert measured_snr_db(clean, noisy) == pytest.approx(snr, abs=0.5)


def test_noise_is_deterministic_per_key_and_differs_across_keys():
    clean = np.random.default_rng(0).normal(size=16000).astype(np.float32)
    assert np.array_equal(add_noise(clean, 10, "a"), add_noise(clean, 10, "a"))
    assert not np.array_equal(add_noise(clean, 10, "a"), add_noise(clean, 10, "b"))


def test_silence_unchanged():
    z = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(add_noise(z, 10, "x"), z)


def test_background_mix_hits_snr_and_is_not_clipped():
    clean = np.full(1000, 0.8, dtype=np.float32)
    noise = np.random.default_rng(2).normal(size=1000).astype(np.float32)
    mixed = mix_background_noise(clean, noise, 5)
    assert measured_snr_db(clean, mixed) == pytest.approx(5, abs=0.01)
    assert np.max(mixed) > 1.0  # preserve the exact requested SNR; do not silently clip


def test_to_mono_16k_passthrough_and_downmix():
    stereo = np.random.default_rng(0).normal(size=(16000, 2)).astype(np.float32)
    mono, sr = to_mono_16k(stereo, 16000)
    assert mono.ndim == 1 and sr == 16000 and len(mono) == 16000
