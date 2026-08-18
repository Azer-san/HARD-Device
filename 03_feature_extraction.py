"""
Phase 3 - Feature Extraction

For every preprocessed audio file:
  - compute a 128x128 mel spectrogram (log + PCEN'd), saved as the primary
    CNN input image
  - compute 13-40 MFCCs as a supplementary feature vector (mean-pooled)

Then split into Train (70%) / Val (15%) / Test (15%) and save as .npy.

Usage:
    python 03_feature_extraction.py --data-dir ../data --out-dir ../data/features
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
N_MFCC = 20  # within the 13-40 range requested


def audio_to_mel_image(audio, sr=SAMPLE_RATE, img_size=IMG_SIZE):
    """128x128 mel spectrogram in dB, PCEN-normalized, resized to img_size x img_size."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=img_size, power=1.0)
    pcen = librosa.pcen(mel * (2 ** 31), sr=sr)

    # Fix time axis to img_size columns (pad or crop)
    if pcen.shape[1] < img_size:
        pad_width = img_size - pcen.shape[1]
        pcen = np.pad(pcen, ((0, 0), (0, pad_width)), mode="constant")
    else:
        pcen = pcen[:, :img_size]

    # Normalize to [0, 1] for CNN input
    pcen = (pcen - pcen.min()) / (pcen.max() - pcen.min() + preprocessing.EPSILON)
    return pcen.astype(np.float32)


def audio_to_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Mean-pooled MFCC vector (supplementary feature, not fed to the CNN by default)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1).astype(np.float32)  # shape: (n_mfcc,)


def build_dataset(data_dir, img_size=IMG_SIZE, n_mfcc=N_MFCC):
    classes = {"no_weevil": 0, "weevil": 1}
    images, mfccs, labels, paths = [], [], [], []

    for cls_name, cls_label in classes.items():
        cls_dir = os.path.join(data_dir, cls_name)
        wav_files = sorted(glob.glob(os.path.join(cls_dir, "*.wav")))
        print(f"  {cls_name}: {len(wav_files)} files")

        for wav_path in wav_files:
            try:
                audio = preprocessing.preprocess_audio(wav_path)
                img = audio_to_mel_image(audio, img_size=img_size)
                mfcc = audio_to_mfcc(audio, n_mfcc=n_mfcc)
            except Exception as e:
                print(f"    skipping {wav_path}: {e}")
                continue

            images.append(img)
            mfccs.append(mfcc)
            labels.append(cls_label)
            paths.append(wav_path)

    X_img = np.array(images, dtype=np.float32)[..., np.newaxis]  # (N, H, W, 1)
    X_mfcc = np.array(mfccs, dtype=np.float32)                    # (N, n_mfcc)
    y = np.array(labels, dtype=np.float32)                        # (N,)
    return X_img, X_mfcc, y, paths


def split_dataset(X_img, X_mfcc, y, train_frac=0.7, val_frac=0.15, seed=42):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    splits = {}
    for name, ix in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = {
            "X_img": X_img[ix],
            "X_mfcc": X_mfcc[ix],
            "y": y[ix],
        }
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "features"))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--n-mfcc", type=int, default=N_MFCC)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Extracting features...")
    X_img, X_mfcc, y, paths = build_dataset(args.data_dir, args.img_size, args.n_mfcc)
    print(f"Total samples: {len(y)}  (weevil={int(y.sum())}, no_weevil={int((1 - y).sum())})")

    if len(y) < 10:
        print("WARNING: very small dataset. Aim for 50+ samples per class before training.")

    print("Splitting into train/val/test (70/15/15)...")
    splits = split_dataset(X_img, X_mfcc, y)

    for name, d in splits.items():
        np.save(os.path.join(args.out_dir, f"X_img_{name}.npy"), d["X_img"])
        np.save(os.path.join(args.out_dir, f"X_mfcc_{name}.npy"), d["X_mfcc"])
        np.save(os.path.join(args.out_dir, f"y_{name}.npy"), d["y"])
        print(f"  {name}: {len(d['y'])} samples -> saved to {args.out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
