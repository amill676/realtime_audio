__author__ = 'adamjmiller'

from pa_tools.stftmanager import StftManager
import numpy as np

dft_len = 8
window_len = 8
hop_len = 4
n_channels = 1

stft = StftManager(dft_length=dft_len, window_length=window_len,
                   hop_length=window_len, # No overlap -- easy to check accuracy
                   use_window_fcn=False,
                   n_channels=n_channels)
data = np.array([1, 0, 2, 0, 1, 1, 1, 0], dtype=np.float32)
stft.performStft(data)

reals, imags = stft.getDFTs()
real_str = ""
imag_str = ""
for arr in reals:
    print arr
for arr in imags:
    print arr

print "reals: " + real_str
print "imags: " + imag_str
print


data1 = np.zeros(8, dtype=np.float32)
stft.performIStft(data1)
print data1


