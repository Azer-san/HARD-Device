"""
Phase 4 - CNN Training

Architecture:
  Input(128,128,1) -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Flatten
  -> Dense(64) -> Dense(1, sigmoid)

Compiles with Adam + binary_crossentropy, trains with early stopping on
val_loss, saves weevil_model.keras plus a confusion matrix and
accuracy/loss curve plots.

Usage:
    python 04_train_model.py --features-dir ../data/features --out-dir ../models
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_model(input_shape=(128, 128, 1)):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_split(features_dir, name):
    X = np.load(os.path.join(features_dir, f"X_img_{name}.npy"))
    y = np.load(os.path.join(features_dir, f"y_{name}.npy"))
    return X, y


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path):
    from collections import Counter
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    labels = ["no_weevil", "weevil"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "features"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train = load_split(args.features_dir, "train")
    X_val, y_val = load_split(args.features_dir, "val")
    X_test, y_test = load_split(args.features_dir, "test")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_model(input_shape=X_train.shape[1:])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    plot_history(history, os.path.join(args.out_dir, "training_curves.png"))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(np.float32)
    plot_confusion_matrix(y_test, y_pred, os.path.join(args.out_dir, "confusion_matrix.png"))

    model_path = os.path.join(args.out_dir, "weevil_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    if test_acc >= args.target_accuracy:
        print(f"✅ Target accuracy ({args.target_accuracy:.0%}) reached — proceed to Phase 5 (TFLite conversion).")
    else:
        print(f"❌ Below target accuracy ({args.target_accuracy:.0%}). Consider:")
        print("   - collecting more/cleaner samples per class")
        print("   - adding data augmentation (time/pitch shift, noise injection)")
        print("   - tuning filters, dropout, learning rate, or trying a deeper CNN")
        print("   - checking for class imbalance or mislabeled clips")


if __name__ == "__main__":
    main()
