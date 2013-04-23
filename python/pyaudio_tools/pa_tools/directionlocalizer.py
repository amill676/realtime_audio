from mattools import mattools as mat

__author__ = 'Adam Miller'
from pa_tools.audiolocalizer import AudioLocalizer
import scipy.fftpack as fftp
import numpy as np
import math
import sys
import constants as consts


class DirectionLocalizer(AudioLocalizer):

    def __init__(self, mic_layout, sample_rate=44100, shift_n=31, shift_max=15):
        """
         :param mic_layout: locations of microphones. Each row should be the
                             location of a given microphone. The dimension
                             is taken to be the number of columns of this
                             matrix
         :type mic_layout: numpy array
         :param sample_rate: Sample rate that was used when sampling data
         :type sample_rate: int
         :param shift_n: Number of points to break shift range into
         :type shift_n: int
         :param shift_max: Will search for shifts in range -shift_max to shift_max
         :type shift_max: int
         """
        AudioLocalizer.__init__(self, mic_layout, sample_rate=sample_rate)
        if mic_layout is not None:
            self._mic_layout = mic_layout.copy()
            self._setup_distances()
        self._shift_n = shift_n + (1 - (shift_n % 2))  # Use odd number so zero shift possible
        self._shift_max = shift_max

    def get_peaks(self, ffts):
        """
        get direction to source
        dfts should be output of getDFTs using no overlap windowing
        """
        #ffts = self.to_matlab_format(dfts)
        dft_len = ffts.shape[1]
        num_chan = ffts.shape[0]
        if len(ffts.shape) > 2:
            ffts = ffts[:, :, 0]
            # Auto correlation in frequency (unshifted)
        auto_corr = ffts[0, :].conjugate() * ffts[1:, :]

        # Discrete freq domain
        k = np.arange(dft_len, dtype=consts.REAL_DTYPE)
        # Shifts
        taus = np.linspace(-self._shift_max, self._shift_max, self._shift_n)
        # Will hold values at t=0 of transformed autocorr
        peaks = np.zeros((num_chan, self._shift_n))
        for idx, tau in enumerate(taus):
            shifted = auto_corr * np.exp(-1j * 2 * np.pi * k * tau / dft_len)
            corr = np.real(fftp.ifft(shifted))
            peaks[0, idx] = tau  # Store shift amount in first row
            peaks[1:, idx] = corr[:, 0]  # Check value at t = 0. Searching for delta
        return peaks

    def get_direction(self, dfts):
        ffts = mat.to_matlab_format(dfts)
        #print "ffts: " + str(ffts)
        return self.get_direction_np(ffts)

    def get_direction_np(self, ffts):
        """
        Determine the direction to the source of given data
        """
        peaks = self.get_peaks(ffts)
        #print "peaks: " + str(peaks)
        samp_period = 1. / float(self._sample_rate)
        delays = np.argmax(peaks, 1)  # Indices in peaks
        #print "delays: " + str(delays)
        delays = peaks[0, delays[1:]] * samp_period * consts.SPEED_OF_SOUND
        #print "delays: " + str(delays)

        if self._use_angle:
            ratio = np.linalg.norm(delays, 2) / np.linalg.norm(self._distances, 2)
            print "ratio : " + str(ratio)
            if abs(abs(ratio) - 1) < 1e-4:  # Ensure magnitude <= 1
                ratio = ratio / abs(ratio)
            angle = math.acos(ratio)

            print "angle: " + str(angle)
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
            direction = rot_mat.dot(self._distances.T)
            return direction.T[0]

        # Now we have the time delays, so we solve system
        direction = mat.cholesky_solve(self._distances, delays)
        norm = np.linalg.norm(direction, 2)
        if norm != 0:
            direction /= np.linalg.norm(direction, 2)
        return direction

    def _setup_distances(self):
        """
        Setup array of distances between mics using the mic
        layout given for this object
        """
        self._use_angle = False  # Whether to use angular approach instead of
        #   least squares
        self._n_mics = self._mic_layout.shape[0]
        self._n_dimensions = self._mic_layout.shape[1]
        if self._n_mics < self._n_dimensions + 1:
            print >> sys.stderr, "WARNING: Should have at least 1 more mic " \
                                 "than number of dimensions in search space"
        if self._n_mics == 2 and self._n_dimensions == 2:
            self._use_angle = True
        if self._n_mics < self._n_dimensions:
            raise ValueError("Must have at least as many mics as dimensions")
            #print "mic layout: " + str(self._mic_layout)
        self._distances = self._mic_layout[1:, :] - self._mic_layout[0, :]