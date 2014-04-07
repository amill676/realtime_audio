__author__ = 'Adam Miller'
import numpy as np
import pa_tools.constants as consts


# Possible improvements: store frequency space, and make user enter
# the fft length at construction

class BeamFormer(object):
    """
    'Abstract' class for performing beam forming on audio signals
    Derived classes should implement a specific type of beam-forming
    """
    def __init__(self, mic_positions, sample_rate):
        """
        :param mic_positions: The positions of each microphone in the microphone
                                array of this beam-former. Each row should
                                represent a different mic, with the number of
                                columns indicating the dimensionality of the space
        :type mic_positions: ndarray
        """
        self._process_mic_positions(mic_positions)
        self._sample_rate = sample_rate

    def _process_mic_positions(self, mic_positions):
        """
        Check the self._mic_positions for validity
        """
        if mic_positions is None:
            raise ValueError("Must supply mic positions for Beam Former")
        self._mic_positions = mic_positions
        self._n_dimensions = self._mic_positions.shape[1]
        self._n_mics = self._mic_positions.shape[0]
        self._mic_distances = self._mic_positions[1:, :] - self._mic_positions[0, :]

    def filter_real(self, all_rffts, align_mat):
        """
        Uses delay and sum technique
        """
        if align_mat.shape != all_rffts.shape[:2]:
            raise ValueError("align_mat and all_rffts first 2 dimensions don't match")
        n_hops = all_rffts.shape[2]
        dft_len = align_mat.shape[1]
        output = np.empty((n_hops, dft_len), dtype=consts.COMPLEX_DTYPE)
        h = self.get_filter()
        for k in range(n_hops):
            output[k, :] = h.T.dot(align_mat * all_rffts[:, :, k])
        return output

    def _get_delays(self, doa):
        """
        Return the sample delays associated with a given doa for the
        microphone array being used by the beamformer. The delay for mic n
        is  defined as the number of samples that will
        the occur between the source signal reaching the first microhpone
        in the array (microphone at row 0 in mic_positions) and reaching
        the nth microhpone. If it is positive, then the signal will
        reach the first microphone before the other, otherwise, the signal
        will reach the first microphone after.

        :param doa: direction of arrival as a ndarray vector. Should be of same
                    dimensionality as the space of the microphones
        :return: delay in samples for each microphone relative to the first
                 microphone (represnted by row 0 in mic_positions)
        """
        mic_0_delay = 0
        distances = np.insert(-1 * self._mic_distances.dot(doa), 0, mic_0_delay)
        return self._sample_rate * distances / consts.SPEED_OF_SOUND

    def get_filter(self):
        """
        Get the spatial filter for given direction of arrival.
        """
        #if len(doa) != self._n_dimensions:
        #    ValueError("The direction of arrival should have the same dimensions"
        #               "as the space in which the microphones are located")
        return np.ones((self._n_mics,)) / self._n_mics  # Delay and sum filter

    def get_beam(self, align_mat, align_mats, rffts, freq):
        """
        Return the spatial filter magnitude response for a given frequency
        corresponding to a source from a direction that will result in the
        given alignment matrix
        :param align_mat: matrix for aligning signal DFTs that will be unique for
                          a certain DOA. Should be in same format as for filter_real.
                          This means it should only have alignment for positive part
                          of the DFT
        :param align_mats: 3 dimensional matrix where each entry along the third dimension
                            is an alignment matrix of the form of align_mat. Each of
                            these alignment matrixes corresponds to the alignment of a possible
                            source direciton
        :param freq: frequency in Hz
        :returns: spatial filter magnitude response for coordinates as returned by
        """
        pos_dft_len = align_mat.shape[1]
        freq_ind = int((freq / self._sample_rate) * (pos_dft_len - 1) * 2)
        h = self.get_filter() #* align_mat[:, freq_ind]
        response = align_mats[:, freq_ind, :].conj()
        # Get first windowed portion at correct frequency
        rfft = np.tile(rffts[:, freq_ind, 0], (response.shape[1], 1)).T
        shifted = response * rfft
        response = h.dot(shifted)
        return np.abs(response)



