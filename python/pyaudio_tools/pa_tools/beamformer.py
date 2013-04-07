__author__ = 'Adam Miller'


class BeamFormer(object):
    """
    'Abstract' class for performing beam forming on audio signals
    Derived classes should implement a specific type of beam-forming
    """
    def __init__(self, mic_positions):
        """
        :param mic_positions: The positions of each microphone in the microphone
                                array of this beam-former. Each row should
                                represent a different mic, with the number of
                                columns indicating the dimensionality of the space
        :type mic_positions: ndarray
        """
        self._process_mic_positions(mic_positions)

    def _process_mic_positions(self, mic_positions):
        """
        Check the self._mic_positions for validity
        """
        if mic_positions is None:
            ValueError("Must supply mic positions for Beam Former")
        self._mic_positions = mic_positions
        self._n_dimensions = self._mic_positions.shape[1]
        self._n_mics = self._mic_positions.shape[0]

    def _get_filter(self, doa):
        """
        Get the spatial filter for given direction of arrival.
        """
        if len(doa) != self._n_dimensions:
            ValueError("The direction of arrival should have the same dimensions"
                       "as the space in which the microphones are located")
        # Get delays for the given doa
        delays = self._mic_positions.dot(doa)

