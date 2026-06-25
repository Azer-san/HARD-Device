import librosa as lb
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
norm_audio = raw_audio / (np.max(np.abs(raw_audio)) + 1e-9)

#pre-emphasis amplifier
  
emp_audio = np.append(norm_audio[:1], norm_audio[1:] - 0.97 * norm_audio[:-1])

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

