import librosa as lb
import librosa.util as lbu
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import os
import scipy as sp
from scipy.signal import butter, filtfilt

#Define paths to file/s
#Add more paths if needed
# format filename = os.path.join("path", "to", "filename.wav")

filename = os.path.join("path", "to", "filename.wav")

#load and convert wav file
raw_audio, sr = lb.load(filename, sr = 22050, mono = True) 

# Normalize amplitude (epsilon guards against silent recordings)
def normalize_audio(x):
    return x / (np.max(np.abs(x)) + 1e-9)

norm_audio = normalize_audio(raw_audio)

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
filt_audio = filter_audio(emp_audio, sr = 22050, lowcut = 1000, highcut = 8000 )

#Audio amplification

def apply_gain(x, gain_factor):
    return  np.clip(x * gain_factor, -1.0, 1.0)

filt_audio = normalize_audio(filt_audio)
amp_audio = apply_gain(filt_audio, gain_factor = 2.0)
peak_audio = amp_audio / np.max(np.abs(amp_audio))

mu = 255
mu_audio = np.sign(peak_audio) * np.log1p(mu * np.abs(peak_audio)) / np.log1p(mu)




