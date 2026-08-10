import librosa as lb
import soundfile as sf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import os
import scipy as sp
import sounddevice as sd
import json 
from scipy.signal import butter, filtfilt



config = {
    "sample_rate": 22050,
    "target_rms": 0.1,
    "lowcut": 2000,
    "highcut": 8000,
    "filter_order": 4,
    "gain_factor": 2.0,
    "compress_threshold": 0.3,
    "compress_ratio": 4.0
}

def record_audio (filename, duration = 5, sr = 44100):
    print(f"Recording {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype = 'int16')
    sd.wait()  # wait for the recording to complete
    sf.write(filename, audio.flatten(), sr) #Save as WAV file
    print(f"Recording saved to {filename}")

if __name__ == "__main__":
    # Record audio and save to file
    record_audio("path/to/filename.wav", duration=10, sr=config["sample_rate"])
    print("Audio recording complete.")

# Normalize amplitude (epsilon guards against silent recordings)
def normalize_audio(x):
    return x / (np.max(np.abs(x)) + 1e-9)

#pre-emphasis amplifier

def pre_emp(x, coeff = 0.97):  
    return np.append(x[:1], x[1:] - coeff * x[:-1])

#filter function

def filter_audio (emp_audio, sr, lowcut, highcut, order = 4):
    nyq = sr / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, emp_audio)
    return y
#May adjust lowcut and highcut depending on the desired frequency(hz) level

#Audio amplification

def apply_gain(x, gain_factor):
    return  np.clip(x * gain_factor, -1.0, 1.0)

def rms_norm(audio, target_rms = config["target_rms"]):
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms == 0:
        return audio  # Avoid division by zero; return original audio if silent
    return audio * (target_rms / current_rms)

def compress(audio, threshold = config["compress_threshold"], ratio = config["compress_ratio"]):
    compressed = audio.copy()
    mask = np.abs(audio) > threshold
    excess = np.abs(audio[mask]) - threshold
    compressed[mask] = np.sign(audio[mask]) * (threshold + excess / ratio)
    return compressed

def pcen(audio, sr, hop_length = 512):
    mel_spec = lb.feature.melspectrogram(y = audio, sr = sr, hop_length = hop_length)
    pcen_spec = lb.pcen(mel_spec * (2**31), sr = sr, hop_length = hop_length)
    return pcen_spec

#Define paths to file/s
#Add more paths if needed
# format filename = os.path.join("path", "to", "filename.wav")

filename = os.path.join("path", "to", "filename.wav")

#load and convert wav file
raw_audio, sr = lb.load(filename, sr = config["sample_rate"], mono = True) 

norm_audio = normalize_audio(raw_audio)

emp_audio = pre_emp(norm_audio)  
filt_audio = filter_audio(emp_audio = emp_audio, sr = config["sample_rate"], lowcut = config["lowcut"], highcut = config["highcut"] )

filt_audio = normalize_audio(filt_audio)
filt_audio = rms_norm(filt_audio)                       # balance energy levels
amp_audio = apply_gain(filt_audio, gain_factor = 2.0)   # boost the signal
cmp_audio = compress(amp_audio)                         # even out quiet vs loud moments






mu = 255
mu_audio = np.sign(cmp_audio) * np.log1p(mu * np.abs(cmp_audio)) / np.log1p(mu)




