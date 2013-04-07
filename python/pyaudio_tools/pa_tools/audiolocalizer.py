__author__ = 'Adam Miller'

import cython
import sys
import numpy as np
import math
import scipy.fftpack as fftp
import scipy.signal as sig
import constants as consts

class AudioLocalizer:

    CUTOFF_FREQ = 2000  # in Hz
    N_ANGLES = 60

    def __init__(self, mic_layout, dft_len=512, sample_rate=44100):
        """
        @type mic_layout: numpy array
        @param mic_layout: a numpy array with each row containing the location
                            of the corresponding microphone. The number of columns
                            of this array will represent the number of dimensions
                            in which the source should be represented. Note that
                            the order of the microphones in this array should
                            correspond with the ordering of the input channel
                            associated with them
        """
        self._sample_rate = float(sample_rate)
        self._dft_len = dft_len
        self._lpf_H = self._create_filter()

    def _create_filter(self):
        """
        Create the low pass filter that will be used to filter the signals
        before performing DOA estimation
        """
        DEFAULT_FILTER_LEN = 32
        cutoff = float(self.CUTOFF_FREQ) / float(self._sample_rate)
        filt = sig.firwin(DEFAULT_FILTER_LEN, cutoff=cutoff)
        return fftp.fft(filt, self._dft_len)


