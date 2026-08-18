"""
Phase 5 - Live Inference on Raspberry Pi

Loop: mic -> record chunk -> preprocess -> extract mel-spectrogram feature
-> TFLite predict -> if confidence > threshold: trigger alert/log detection.

Designed to run with `tflite-runtime` (lightweight, ARM-friendly) but falls
back to full `tensorflow` if that's what's installed.

Usage (on the Pi):
    python 06_live_inference.py --model ../models/weevil_model.tflite \
        --threshold 0.7 --chunk-duration 2.0
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(__file__))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "preprocessing", os.path.join(os.path.dirname(__file__), "02_preprocessing.py"))
preprocessing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocessing)

_spec2 = importlib.util.spec_from_file_location(
    "feature_extraction", os.path.join(os.path.dirname(__file__), "03_feature_extraction.py"))
feature_extraction = importlib.util.module_from_spec(_spec2)
# feature_extraction.py's own module-level import of 02_preprocessing runs fine standalone
_spec2.loader.exec_module(feature_extraction)

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("tflite_runtime not found, falling back to tensorflow.lite")
    import tensorflow as tf
    tflite = tf.lite


class WeevilDetector:
    def __init__(self, model_path, threshold=0.7):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.threshold = threshold

        # INT8 quantization params (if quantized model)
        self.input_scale, self.input_zero_point = self.input_details[0].get(
            "quantization", (0.0, 0))
        self.output_scale, self.output_zero_point = self.output_details[0].get(
            "quantization", (0.0, 0))
        self.is_quantized = self.input_details[0]["dtype"] == np.int8

    def predict(self, mel_image):
        """mel_image: (128, 128) float32 in [0, 1]."""
        x = mel_image[np.newaxis, ..., np.newaxis]  # (1, 128, 128, 1)

        if self.is_quantized:
            x = (x / self.input_scale + self.input_zero_point).astype(np.int8)
        else:
            x = x.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])

        if self.is_quantized:
            out = (out.astype(np.float32) - self.output_zero_point) * self.output_scale

        return float(out.ravel()[0])

    def is_weevil(self, mel_image):
        prob = self.predict(mel_image)
        return prob > self.threshold, prob


def record_chunk(duration, sr):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.ravel()


def log_detection(prob, log_path):
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts},weevil_detected,confidence={prob:.4f}\n"
    with open(log_path, "a") as f:
        f.write(line)
    print(f"🐛 WEEVIL DETECTED  (confidence={prob:.2%})  @ {ts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "models", "weevil_model.tflite"))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--chunk-duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=preprocessing.SAMPLE_RATE)
    parser.add_argument("--log-path", default=os.path.join(os.path.dirname(__file__), "..", "detections.csv"))
    parser.add_argument("--test-file", default=None,
                         help="Run once on a single .wav file instead of live mic loop (for pre-deployment testing)")
    args = parser.parse_args()

    detector = WeevilDetector(args.model, threshold=args.threshold)

    if args.test_file:
        audio = preprocessing.preprocess_audio(args.test_file, sr=args.sample_rate)
        mel_img = feature_extraction.audio_to_mel_image(audio, sr=args.sample_rate)
        detected, prob = detector.is_weevil(mel_img)
        print(f"File: {args.test_file}")
        print(f"Prediction: {'WEEVIL' if detected else 'no weevil'}  (confidence={prob:.2%})")
        return

    print(f"Starting live inference loop. Threshold={args.threshold}, "
          f"chunk={args.chunk_duration}s. Ctrl+C to stop.")

    try:
        while True:
            raw_audio = record_chunk(args.chunk_duration, args.sample_rate)

            # run the same preprocessing chain used at training time
            audio = raw_audio.astype(np.float32)
            peak = np.max(np.abs(audio))
            audio = audio / (peak + preprocessing.EPSILON)
            audio = preprocessing.pre_emphasis(audio)
            audio = preprocessing.bandpass_filter(audio, sr=args.sample_rate)
            audio = preprocessing.rms_normalize(audio)
            audio = preprocessing.gain_boost(audio)
            audio = preprocessing.dynamic_range_compression(audio)
            audio = preprocessing.mu_law_encode(audio)

            mel_img = feature_extraction.audio_to_mel_image(audio, sr=args.sample_rate)
            detected, prob = detector.is_weevil(mel_img)

            if detected:
                log_detection(prob, args.log_path)
            else:
                print(f"  ... no weevil (confidence={prob:.2%})")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
