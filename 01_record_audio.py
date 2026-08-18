"""
Phase 1 - Data Collection
Record weevil / background audio samples on the Raspberry Pi mic.

Usage:
    python 01_record_audio.py --label weevil --count 50
    python 01_record_audio.py --label no_weevil --count 50

Each run records `count` clips of `duration` seconds each, saved as
data/<label>/<label>_<index>_<timestamp>.wav

Press Ctrl+C at any time to stop early (already-saved clips are kept).
"""

import argparse
import json
import os
import time
from datetime import datetime

import sounddevice as sd
from scipy.io.wavfile import write as wav_write

SAMPLE_RATE = 22050          # match the rate used in preprocessing
CHANNELS = 1
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def record_clip(duration, sample_rate=SAMPLE_RATE, channels=CHANNELS):
    print(f"  recording {duration:.1f}s ...", end="", flush=True)
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                    channels=channels, dtype="int16")
    sd.wait()
    print(" done")
    return audio


def main():
    parser = argparse.ArgumentParser(description="Record labelled audio samples")
    parser.add_argument("--label", required=True, choices=["weevil", "no_weevil"],
                         help="Class to record")
    parser.add_argument("--count", type=int, default=50, help="Number of clips to record")
    parser.add_argument("--duration", type=float, default=2.0, help="Clip length in seconds")
    parser.add_argument("--pause", type=float, default=1.0,
                         help="Seconds to pause between clips (lets you reposition mic)")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    args = parser.parse_args()

    out_dir = os.path.join(DATA_ROOT, args.label)
    os.makedirs(out_dir, exist_ok=True)

    existing = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
    start_index = len(existing)

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    print(f"Recording {args.count} '{args.label}' clips "
          f"({args.duration}s each, {args.sample_rate} Hz). Starting at index {start_index}.")
    print("Ctrl+C to stop early.\n")

    try:
        for i in range(args.count):
            idx = start_index + i
            print(f"[{i + 1}/{args.count}] clip #{idx}")
            audio = record_clip(args.duration, args.sample_rate)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{args.label}_{idx:04d}_{ts}.wav"
            fpath = os.path.join(out_dir, fname)
            wav_write(fpath, args.sample_rate, audio)

            manifest.append({
                "file": fname,
                "label": args.label,
                "sample_rate": args.sample_rate,
                "duration_sec": args.duration,
                "recorded_at": ts,
            })

            if i < args.count - 1:
                time.sleep(args.pause)

    except KeyboardInterrupt:
        print("\nStopped early by user.")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {len(manifest)} total clips to {out_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
