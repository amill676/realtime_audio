from mattools import mattools as mat

__author__ = 'Adam Miller'
from pa_tools.audiolocalizer import AudioLocalizer
import numpy as np
import math
import constants as consts
import sys

class DistributionLocalizer(AudioLocalizer):

    def __init__(self, mic_positions, dft_len=512, sample_rate=44100, n_theta=20, n_phi=20):
        """
        :param mic_positions: locations of microphones. Each row should be the
                            location of a given microphone. The dimension
                            is taken to be the number of columns of this
                            matrix
        :type mic_positions: numpy array
        :param sample_rate: Sample rate that was used when sampling data
        :type sample_rate: int
        :param n_theta: The number of points to sample in theta search
                        space where theta is angle in spherical coordinates
        :type n_theta: int
        """
        AudioLocalizer.__init__(self, mic_positions, dft_len=dft_len, sample_rate=sample_rate)
        self._n_theta = n_theta
        self._n_phi = n_phi
        if mic_positions is not None:
            self._mic_positions = mic_positions.copy()
            self._process_mic_positions()
            self._setup_distances()
            self._setup_search_space()
            self._setup_freq_spaces()
        else:
            sys.stderr.write("WARNING: No mic positions provided -- certain public methods"
                             "may fail")

    def get_distribution_mat(self, ffts):
        """
        Get probability distribution of source location
        :param ffts:
        """
        #energies = ffts * ffts.conjugate()
        #THRESHOLD = 180
        #energies = np.sum(energies, axis=1)
        #total_energy = np.sum(energies)
        #if total_energy < THRESHOLD:
        #    return self._prev_distr
        #if self._dft_len != ffts.shape[1]:
        #    raise ValueError("DFT's given do not have same size as that specified at construction")
        #ffts *= self._lpf_H  # LPF incoming audio
        ffts[:, self._cutoff_index:-self._cutoff_index + 1] = 0
        auto_corr = ffts[0, :] * ffts[1:, :].conjugate()
        auto_corr /= (np.abs(ffts[0, :]) + consts.EPS)
        # For some reason, 'whitening' both sources causes problems (underflow?)
        #auto_corr /= (np.abs(ffts[0, :]) * np.abs(ffts[1:, :]))
        # Get correlation values
        corrs = np.zeros((self._n_mics - 1, self._n_theta * self._n_phi), dtype=consts.REAL_DTYPE)
        for i in range(corrs.shape[1]):
            shifted = auto_corr * self._shift_mats[:, :, i]
            corrs[:, i] = np.real(np.sum(shifted, axis=1))  # idft for n = 0
        # Normalize ang get probability for each direction
        distr = np.empty((4, self._n_theta * self._n_phi), dtype=consts.REAL_DTYPE)
        distr = np.maximum(np.prod(corrs, axis=0), consts.EPS)
        #distr[3, :] = np.log(distr[3, :])
        #distr[3, :] -= np.log(1.e14)
        return distr

    def get_distribution_real(self, rffts):
        cutoff_index = (self.CUTOFF_FREQ / self._sample_rate) * (self._dft_len / 2)
        lowffts = rffts[:, :cutoff_index]  # Low pass filtered
        auto_corr = lowffts[0, :] * lowffts[1:, :].conjugate()
        auto_corr /= (np.abs(lowffts[0, :]) + consts.EPS)
        # Get correlation values from time domain
        corrs = np.zeros((self._n_mics - 1, self._n_theta * self._n_phi), dtype=consts.REAL_DTYPE)
        for i in range(corrs.shape[1]):
            shifted = auto_corr * self._lp_pos_shift_mats[:, :, i]
            corrs[:, i] = np.real(shifted[:, 0] + 2 * np.sum(shifted[:, 1:], axis=1))  # ifft for n = 0
        distr = np.maximum(np.prod(corrs, axis=0), consts.EPS)
        return distr

    def get_3d_real_distribution(self, dfts):
        rffts = mat.to_real_matlab_format(dfts)
        return self.get_distribution_real(rffts)

    def get_3d_distribution(self, dfts):
        """
        Get distribution of source direction across 3d search space
        """
        ffts = mat.to_matlab_format(dfts)
        return self.get_distribution_mat(ffts)

    def get_spher_directions(self):
        """
        Returns the spherical coordinates that correspond to the same
        cartesian coordinates that are returned from get_3d_distribution
        The row 0 will be the radius (all 1's - unit vectors), the
        row 1 will be the azimuthal angle, and row 2 will be the polar angle
        """
        return self._spher_directions.copy()

    def get_directions(self):
        """
        Returns the array of source directions that is searched when doing
        DOA estimation. The format and indices of the entries coincide with
        many of the other available methods for this module
        """
        return self._directions.copy()

    def to_spher_grid(self, distr):
        """
        Will return the distrbution array as a grid over spherical coords.
        This way, the grid can be used for plotting surfaces with meshed
        spherical coordinates. Axis 0 will vary with polar angle and
        axis 1 will vary with azimuthal angle
        :param distr: array corresponding to the distribution given by a
                      call to get_3d_distribution
        :type distr: numpy array (vector)
        """
        return np.reshape(distr, (self._n_phi, self._n_theta))

    def get_align_mat(self):
        """
        This will return the set of matrices that, when multiplied by
        the FFTs of each microhpones signal, will align the signals
        as if they came from the different possible source directions.

        The returned matrix is of size (nmics, dft_len, n_phi * n_theta)
        Therefore, for a returned matrix 'mat', mat[:, :, i] will give
        the described alignment matrix for a source coming from direction
        (get_directions)[:, i]
        """
        sm_copy = self._shift_mats.copy()
        mic0_shifts = np.ones((1, self._dft_len, self._n_theta * self._n_phi))
        return np.vstack((mic0_shifts, sm_copy))

    def get_pos_align_mat(self):
        """
        This will return the alignment matrices as described in get_align_mat()
        This version will return only the alignments corresponding
        to the positive frequencies and negative nyquist frequency, which
        assumes that the signals are real.
        """
        sm_copy = self._pos_shift_mats.copy()
        mic0_shifts = np.ones((1, self._dft_len / 2 + 1, self._n_theta * self._n_phi))
        return np.vstack((mic0_shifts, sm_copy))

    def _process_mic_positions(self):
        self._n_mics = self._mic_positions.shape[0]
        self._n_dimensions = self._mic_positions.shape[1]
        if self._n_dimensions != 3:
            ValueError("Microphones must be specified in 3 dimensions")

    def _setup_distances(self):
        """
        Setup array of distances between mics using the mic
        layout given for this object
        """
        self._distances = self._mic_positions[1:, :] - self._mic_positions[0, :]

    def _setup_search_space(self):
        """
        Setup the search space for constructing the distribution of source
        locations. This will be held in the member variable 'directions'.
        This method will also setup the member variable 'delays', which
        will contain the delays in samples between the first microphone and
        every other microphone. Note that these sample delays may be non-integers.
        """
        # Setup angle space
        theta = np.linspace(0, 2 * math.pi, self._n_theta)
        #theta = theta[:-1]  # Don't use both 0 and 2pi
        phi = np.linspace(0, math.pi / 2, self._n_phi)

        # Setup array of direction vectors
        self._directions = np.empty((3, self._n_theta * self._n_phi), dtype=consts.REAL_DTYPE)
        self._spher_directions = np.empty((3, self._n_theta * self._n_phi), dtype=consts.REAL_DTYPE)
        for p in range(self._n_phi):
            for t in range(self._n_theta):
                ind = p * self._n_theta + t
                # Get cartestian coordinates
                self._directions[0, ind] = np.sin(phi[p]) * np.cos(theta[t])  # x
                self._directions[1, ind] = np.sin(phi[p]) * np.sin(theta[t])  # y
                self._directions[2, ind] = np.cos(phi[p])  # z
                # Get the spherical coordinates
                self._spher_directions[0, ind] = 1
                self._spher_directions[1, ind] = theta[t]
                self._spher_directions[2, ind] = phi[p]

    def _setup_freq_spaces(self):
        # Setup delays between first mic and all others
        self._delays = -1 * self._distances.dot(self._directions) * self._sample_rate / consts.SPEED_OF_SOUND
        # Setup the various shift matrices
        self._setup_shift_mats()
        self._setup_pos_shift_mats()
        self._setup_lp_pos_shift_mats()
        # Store previous distribution for when signal energy is very low
        self._prev_distr = np.zeros((4, self._n_phi * self._n_theta), dtype=consts.REAL_DTYPE)
        self._prev_distr[:3, :] = self._directions

    def _setup_lp_pos_shift_mats(self):
        self._cutoff_index = self._compute_cutoff_index()
        self._lp_pos_shift_mats = np.empty((self._n_mics - 1, self._cutoff_index, self._n_theta * self._n_phi),
                                           dtype=consts.COMPLEX_DTYPE)
        nn = np.arange(0, self._cutoff_index, dtype=consts.REAL_DTYPE)
        if self._cutoff_index == self._dft_len / 2 + 1:
            nn[-1] *= -1  # Last entry is for -(dft_len / 2)
        for i in range(self._lp_pos_shift_mats.shape[2]):
            freqs = np.outer(self._delays[:, i], nn)
            self._lp_pos_shift_mats[:, :, i] = np.exp(-1j * 2 * math.pi * freqs / self._dft_len)

    def _compute_cutoff_index(self):
        """
        Compute the index corresponding to the cutoff frequency for the
        low pass filter being used. Then only the DFT coefficients for indices
        up to this cutoff index should be used in localization
        :returns: cutoff index as described. Will be of int type
        """
        cutoff_index = int((float(self.CUTOFF_FREQ) / self._sample_rate) * (self._dft_len / 2))
        if cutoff_index > self._dft_len / 2 + 1:
            cutoff_index = int(self._dft_len / 2 + 1)
        return cutoff_index

    def _setup_pos_shift_mats(self):
        """
        Setup matrix that can be used to shift ffts to delays corresponding
        with the search space. This will use only the positive frequencies
        and negative nyquist frequency
        """
        self._pos_shift_mats = np.empty(
            (self._n_mics - 1, self._dft_len / 2 + 1, self._n_theta * self._n_phi),
            dtype=consts.COMPLEX_DTYPE)
        nn = np.arange(0, self._dft_len / 2 + 1)
        nn[-1] *= -1  # Negative nyquist freq
        for i in range(self._pos_shift_mats.shape[2]):
            freqs = np.outer(self._delays[:, i], nn)
            self._pos_shift_mats[:, :, i] = np.exp(-1j * 2 * math.pi * freqs / self._dft_len)

    def _setup_shift_mats(self):
        self._shift_mats = np.empty((self._n_mics - 1, self._dft_len, self._n_theta * self._n_phi),
                                    dtype=consts.COMPLEX_DTYPE)
        nn = np.hstack((np.arange(0, self._dft_len / 2, dtype=consts.REAL_DTYPE),
                        np.arange(-self._dft_len / 2, 0, dtype=consts.REAL_DTYPE)))
        for i in range(self._shift_mats.shape[2]):
            freqs = np.outer(self._delays[:, i], nn)
            self._shift_mats[:, :, i] = np.exp(-1j * 2 * math.pi * freqs / self._dft_len)



