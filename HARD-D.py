import librosa as lb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import os

#Define paths to file/s
#Add more paths if needed
# format filename = os.path.join("path", "to", "filename.wav")

filename = os.path.join("path", "to", "filename.wav")
