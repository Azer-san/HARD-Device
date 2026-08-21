"""
Phase 2 - Preprocessing Pipeline

Pipeline per file:
  load -> resample 22050 Hz -> mono -> peak normalize (epsilon guard)
  -> pre-emphasis -> Butterworth bandpass (2000-8000 Hz, order=4, filtfilt)
  -> RMS normalize (target=0.1) -> gain boost -> dynamic range compression
  -> mu-law encoding -> PCEN

This module is imported by both the training pipeline (03_feature_extraction.py)
and the on-device live inference script, so the exact same preprocessing is
applied at train and inference time.
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt

SAMPLE_RATE = 22050
EPSILON = 1e-9

BANDPASS_LOW = 2000
BANDPASS_HIGH = 8000
BANDPASS_ORDER = 4

RMS_TARGET = 0.1
GAIN_BOOST_DB = 6.0
COMPRESSION_THRESHOLD = 0.3
COMPRESSION_RATIO = 4.0
MU_LAW_MU = 255


def load_and_convert(path, sr=SAMPLE_RATE):
    """Load a wav file, resample, force mono, peak normalize."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    peak = np.max(np.abs(audio))
    audio = audio / (peak + EPSILON)
    return audio.astype(np.float32)


def pre_emphasis(audio, coeff=0.97):
    """Boost high frequencies: y[n] = x[n] - coeff * x[n-1]."""
    return np.append(audio[:1], audio[1:] - coeff * audio[:-1])


def bandpass_filter(audio, sr=SAMPLE_RATE, low=BANDPASS_LOW, high=BANDPASS_HIGH,
                     order=BANDPASS_ORDER):
    """Butterworth bandpass, zero-phase (filtfilt) to isolate weevil frequency band."""
    nyquist = 0.5 * sr
    low_norm = low / nyquist
    high_norm = min(high / nyquist, 0.999)  # guard against >= Nyquist
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, audio).astype(np.float32)


def rms_normalize(audio, target=RMS_TARGET):
    rms = np.sqrt(np.mean(audio ** 2)) + EPSILON
    return audio * (target / rms)


def gain_boost(audio, db=GAIN_BOOST_DB):
    factor = 10 ** (db / 20)
    return audio * factor


def dynamic_range_compression(audio, threshold=COMPRESSION_THRESHOLD, ratio=COMPRESSION_RATIO):
    """Simple soft-knee-free compressor applied sample-wise on the envelope sign."""
    sign = np.sign(audio)
    mag = np.abs(audio)
    compressed = np.where(
        mag > threshold,
        threshold + (mag - threshold) / ratio,
        mag,
    )
    return sign * compressed


def mu_law_encode(audio, mu=MU_LAW_MU):
    """Mu-law companding, output kept in float [-1, 1] (not quantized to int)."""
    audio = np.clip(audio, -1.0, 1.0)
    return np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)


def apply_pcen(audio, sr=SAMPLE_RATE):
    """Per-Channel Energy Normalization, applied on a mel spectrogram then
    inverted back to a 1D representation isn't standard - PCEN is normally
    applied directly to the spectrogram (see 03_feature_extraction.py).
    Here we expose it for cases where you want a PCEN'd spectrogram from
    raw audio directly."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, power=1.0)
    pcen = librosa.pcen(mel * (2 ** 31), sr=sr)
    return pcen


def preprocess_audio(path, sr=SAMPLE_RATE):
    """Full preprocessing chain, audio-domain steps only (stops before
    spectrogram/PCEN, which 03_feature_extraction.py handles)."""
    audio = load_and_convert(path, sr)
    audio = pre_emphasis(audio)
    audio = bandpass_filter(audio, sr)
    audio = rms_normalize(audio)
    audio = gain_boost(audio)
    audio = dynamic_range_compression(audio)
    audio = mu_law_encode(audio)
    return audio


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python 02_preprocessing.py <path_to_wav>")
        sys.exit(1)

    out = preprocess_audio(sys.argv[1])
    print(f"Preprocessed audio shape: {out.shape}, "
          f"min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
