"""
Phase 3 - Feature Extraction

For every preprocessed audio file:
  - compute a 128x128 mel spectrogram (log + PCEN'd), saved as the primary
    CNN input image
  - compute 13-40 MFCCs as a supplementary feature vector (mean-pooled)

Then split into Train (70%) / Val (15%) / Test (15%) and save as .npy.

Usage:
    python 03_feature_extraction.py --data-dir data --out-dir data/features
"""

import argparse
import os
import glob
import numpy as np
import librosa
import sys
import importlib.util


sys.path.insert(0, os.path.dirname(__file__))
# workaround: filename starts with a digit so it can't be `import`ed normally
_spec = importlib.util.spec_from_file_location(
    "preprocessing", os.path.join(os.path.dirname(__file__), "02_preprocessing.py"))
preprocessing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocessing)

SAMPLE_RATE = preprocessing.SAMPLE_RATE
IMG_SIZE = 128
N_MFCC = 40  # within the 13-40 range requested


def extract_svm_features(audio, sr=SAMPLE_RATE):
    #Convert audio to 1D for SVM

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)  # shape (40,)
    mfcc_std = np.std(mfcc, axis=1)    # shape (40,)

    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    #combine everything into a single feature vector
    return np.concatenate([mfcc_mean, mfcc_std, [centroid, zcr, rms]])
    # final shape: (83,) — 40 + 40 + 3

def build_dataset(data_dir):
    #Loop weevil/ and no_weevil/ folders, extract features, assign labels
    X, y = [], []
    classes = {"no_weevil": 0, "weevil":1}

    for class_name, label in classes.items():
        folder = os.path.join(data_dir, class_name)
        wav_files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        print(f" {class_name}: {len(wav_files)} files")

        for wav_path in wav_files:
            try:
                audio = preprocessing.preprocess_audio(wav_path)
                features = extract_svm_features(audio)

            except Exception as e:
                print(f"  skipping {wav_path}: {e}")
                continue

            X.append(features)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    # X shape: (N, 83) — one row per audio clip
    # y shape: (N,)   — 0 or 1 per clip


def split_dataset(X, y, train_frac=0.70, val_frac=0.15, seed=42):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])

def main():
    DATA_DIR = "data"
    OUT_DIR = "data/features"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Extracting features...")
    X, y = build_dataset(DATA_DIR)
    print(f"Total samples: {len(y)}  (weevil={int(y.sum())} | no_weevil={int((1 - y).sum())})")

    if len(y) < 20:
        print("WARNING: very small dataset. Aim for atleast 20 samples per class before training.")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    for name, arr in [("X_train", X_train), ("y_train", y_train), 
                    ("X_val", X_val), ("y_val", y_val),
                    ("X_test", X_test), ("y_test", y_test)]:
        np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)
        print(f"  saved{name}.npy - shape {arr.shape}")


if __name__ == "__main__":
    main()
