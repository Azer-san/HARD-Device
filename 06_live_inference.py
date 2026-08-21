#Phase 5 - Live inference on raspberry pi

import os
import sys
import time
import numpy as np
import sounddevice as sd
import joblib
import importlib.util
from datetime import datetime

# --- Constants ---
MODEL_PATH     = "../models/weevil_svm.pkl"
SCALER_PATH    = "../models/scaler.pkl"
THRESHOLD      = 0.7    # confidence needed to trigger detection
CHUNK_DURATION = 2.0    # seconds per recording chunk
LOG_PATH       = "../detections.csv"

#Import from phase 2
_spec = importlib.util.spec_from_location(
    "preprocessing",
    os.path.join(os.path.dirname(__file__), "02_preprocessing.py"))

preprocessing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocessing)

SAMPLE_RATE = preprocessing.SAMPLE_RATE

#Import form phase 3
_spec2 = importlib.util.spec_from_location(
    "feature_extraction",
    os.path.join(os.path.dirname(__file__), "03_feature_extraction.py"))
feature_extraction = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(feature_extraction)

#Load SVM and Scaler
svm = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model and scaler loaded successfully.")

def predict(audio):
    """Run full feature extraction + SVM prediction on one audio chunk."""
    # Step 1 — extract 1D feature vector (shape: (83,))
    features = feature_extraction.extract_features(audio, sr=SAMPLE_RATE)

    #Step 2 - sclae using the same scaler from training
    features_scaled = scaler.transofm(features.reshape(-1, 1))

    #Step 3 - get weevil probability (index 1 = weevil class)
    prob = svm.predict_proba(features_scaled)[0][1]
    return prob

def record_chunk():
    """Record a chunk of audio from the microphone."""
    
    audio = sd.rec(
        int(CHUNK_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is finished
    return audio.flatten() # flatten from (N,1) to (N,)

def preprocess(audio):
    """Run the full preprocessing chain from phase 2"""
    peak = np.max(npabs(audio))
    audio = audio / (peak + preprocessing.EPSILON)  # Normalize to [-1, 1]
    audio = preprocessing.pre_emphaisis(audio)
    audio = preprocessing.bandpass_filter(audio, sr=SAMPLE_RATE)
    audio = preprocessing.rms_normalize(audio)
    audio = preprocessing.gain_boost(audio)
    audio = preprocessing.dynamic_range_compression(audio)
    audio = preprocessing.mu_law_encode(audio)
    return audio

def log_detection(prob):
    """Append detection to CSV with timestamp"""
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts}, weevil_detected, confidence={prob:.4f}\n
    with open(LOG_PATH, "a") as f:
        f.write(line)
    print(f"WEEVIL DETECTED conficence={prob:.2%} @ {ts}")

def test_file(path):
    """Test on a saved .wav file before going live."""
    audio = preprocessing.preprocess_audio(path)
    prob = predict(audio)
    label = "WEEVIL" if prob > THRESHOLD else "NO WEEVIL"
    print(f"file : {path}")
    print(f"Result: {label} (confidence={prob:.2%})")

def live_loop():
    """Continuously record and classify until Ctrl+C is pressed."""
    print(f"Live inference started.")
    print(f"Thresholf = {THRESHOLD} | Chunk = {CHUNK_DURATION}s | Ctrl+C tp stop\n")
    try:
        while True:
            raw = record_chunk()
            audio = preprocess(raw)
            prob = predict(audio)

            if prob > THRESHOLD:
                log_detection(prob)

            else:
                print(f"No weevil detected (confidence={prob:.2%})")
    except KeyboardInterrupt:
        print("\nLive inference stopped.")

# ---Entry point ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        # test mode: python 06_live_inference.py path/to/file.wav
        test_file(sys.argv[1])
    else:
        # live mode: python 06_live_inference.py
        live_loop()
        





if __name__ == "__main__":
    main()
