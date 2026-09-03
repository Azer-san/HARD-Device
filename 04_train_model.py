#Phase 4 - SVM training

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# --- Constants ---
FEATURES_DIR = "data/features"
OUT_DIR      = "models"
TARGET_ACC   = 0.90

def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["No Weevil", "Weevil"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")

def main ():    
    os.makedirs(OUT_DIR, exist_ok=True)


    #Load splits from phase 3

    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    print(f"{X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    scaler = StandardScaler()
    X_trainval_sc = scaler.fit_transform(X_trainval)
    X_test_sc = scaler.transform(X_test)


    #Train SVM model
    print("Training SVM...")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    svm.fit(X_trainval_sc, y_trainval)
    print("Training complete.")

    #Evaluate on the test
    y_pred = svm.predict(X_test_sc)
    y_pred_prob = svm.predict_proba(X_test_sc)[:, 1] #Probability of weevil
    test_acc = np.mean(y_pred == y_test)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred, 
        target_names=["No Weevil", "Weevil"]))


    #Plot points
    plot_confusion_matrix(y_test, y_pred,
        os.path.join(OUT_DIR, "confusion_matrix.png"))

    #Save model and scaler
    joblib.dump(svm, os.path.join(OUT_DIR, "weevil_svm.pkl"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    print("Model and scaler saved.")

    #Check accuracy
    if test_acc >= TARGET_ACC:
        print("Target reached - you may now proceed")

    else:
        print("Below target. Try:")
        print("- More audio samples per class")
        print("- Increase C (e.g. C=10) for a stricter boundary")
        print("- Decrease C (e.g. C=0.1) if overfitting")


# Cell	Name	          Ideal
#     TN	        Correct — no weevil	High
#     TP	        Correct — weevil caught	High
#     FP	        False alarm	Low
#     FN	        Missed weevil	Zero if possible


if __name__ == "__main__":
    main()
