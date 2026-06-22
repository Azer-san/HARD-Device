import librosa as lb
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
audio, sr = lb.load(filename, sr = 22050, mono = True) 

# Normalize amplitude (epsilon guards against silent recordings)
audio = audio / (np.max(np.abs(audio)) + 1e-9)

#pre-emphasis amplifier
  
audio = np.append(audio[:1], audio[1:] - 0.97 * audio[:-1])

#filter function

def filter_audio (audio, sr, lowcut, highcut, order = 4):
    nyq = sr / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, audio)
    return y

audio = filter_audio(audio, sr = 22050, lowcut = 1000, highcut = 8000 )
