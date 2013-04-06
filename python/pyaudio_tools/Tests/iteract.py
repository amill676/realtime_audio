__author__ = 'adamjmiller'

from pa_tools.stftmanager import StftManager
import numpy as np
import scipy.fftpack as dft

dft_len = 8
window_len = 8
hop_len = 4
n_channels = 1

stft = StftManager(dft_length=dft_len, window_length=window_len,
                   hop_length=window_len,  # No overlap -- easy to check accuracy
                   use_window_fcn=False,
                   n_channels=n_channels)
data = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
#data = np.ndarray([window_len, 1], dtype=np.float32)
#data[:, 0] = np.array([1, 0, 1, 0, 1, 0, 1, 0])
stft.performStft(data)
print data.shape

npdft = dft.rfft(data)
print "npdft: " + str(npdft)

dfts = stft.getDFTs() # List of dfts for each channel
reals, imags = dfts[0]

real_part = reals[0]
imag_part = imags[0]

real_str = ""
imag_str = ""
for i in real_part:
    real_str += "  " + str(i)
for i in imag_part:
    imag_str += "  " + str(i)
print "reals: " + real_str
print "imags: " + imag_str

data1 = stft.performIStft()
print "istft: " + str(data1)


