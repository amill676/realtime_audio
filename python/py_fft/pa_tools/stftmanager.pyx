__author__ = 'adamjmiller'

cimport cstftmanager as cstft
import numpy as np
cimport numpy as cnp  # Get declarations in numpy.pxd
cnp.import_array()

cdef class StftManager:
    """
    Wrapper object for c realtimestft library

    NOTES:
        - For the time being, this class does not allow dft's of length
        greater than the window size. This likely relates to some
        implementation chosen in the C code. I'll have to check it out...
    """
    cdef cstft.realtimeSTFT _c_stft
    cdef int _dft_length
    cdef int _window_length
    cdef int _hop_length
    cdef int _n_channels

    def __init__(self,
                  dft_length=1024,
                  window_length=1024,
                  hop_length=512,
                  n_channels=1,
                  use_window_fcn=True,
                  dtype=np.float32):
        # Convert into log parameters for C stft library
        if not self._is_power_of_2(dft_length) or not self._is_power_of_2(window_length) or \
                not self._is_power_of_2(hop_length):
            raise ValueError("StftManager: all input parameters must be powers of 2")

        # Now convert to logs
        cdef int log_dft_length = self._log2(dft_length)
        cdef int log_window_length = self._log2(window_length)
        cdef int log_hop_length = self._log2(hop_length)

        # Check for valid dtype
        cdef int data_size
        if np.dtype(dtype) == np.dtype('float32'):
            data_size = 4
        elif np.dtype(dtype) == np.dtype('float64'):
            data_size = 8
        elif np.dtype(dtype) == np.dtype('float'):
            data_size = 8
        else:
            data_size = -1  # This way an error will occur

        # Check for valid use_window parameter
        cdef int c_use_window_fcn = 1
        if use_window_fcn == True:
            c_use_window_fcn = 1
        elif use_window_fcn == False:
            c_use_window_fcn = 0
        else:
            raise ValueError("use_window_fcn should be of boolean type.")

        # Note that self is not fully constructed at this point, so
        # don't do anything to self but assign cdef fields for now
        error = cstft.createRealtimeSTFT(&self._c_stft,
                                         log_dft_length,
                                         log_window_length,
                                         log_hop_length,
                                         n_channels,
                                         c_use_window_fcn,
                                         data_size
                                         )
        self._check_error(error)
        if &self._c_stft is NULL:
            raise MemoryError("Creation of StftManager failed.")

        # Set member variables
        self._dft_length = dft_length
        self._window_length = window_length
        self._hop_length = hop_length
        self._n_channels = n_channels


    def __dealloc__(self):
        if &self._c_stft is not NULL:
            cstft.destroyRealtimeSTFT(&self._c_stft)

    cdef _check_error(self, cstft.stft_error error):
        """
        Raise the correct exception corresponding to an error code given
        @param error: the error code returned from a call to the
                      realtimestft library
        """
        if error == cstft.STFT_FAILED_MALLOC:
            raise MemoryError("StftManager: failed malloc operation.")
        if error == cstft.STFT_INVALID_HOPSIZE:
            raise ValueError("StftManager: invalid hopsize.")
        if error == cstft.STFT_INVALID_WINDOWSIZE:
            raise ValueError("StftManager: invalid windowsize.")
        if error == cstft.STFT_INVALID_DFTLEN:
            raise ValueError("StftManager: invalid DFT length.")
        if error == cstft.STFT_INVALID_NUM_CHANNELS:
            raise ValueError("StftManager: invalid number of channels.")
        if error == cstft.STFT_NULL_PARAMETER:
            raise ValueError("StftManager: NULL parameter given.")
        if error == cstft.STFT_FFTSETUP_ERROR:
            raise ValueError("StftManager: Error in setting up FFT.")
        if error == cstft.STFT_INVALID_DATA_SIZE:
            raise ValueError("StftManager: Error in data size." +
                             " Should be 4 or 8 for float32 or float64")

    cdef bint _is_power_of_2(self, int n):
        """
        Check if a number is a power of 2
        @return: True if power of 2. False else
        """
        return n != 0 and (n & (n - 1) == 0)

    cdef int _log2(self, int n):
        """
        Get log base 2 of n. Note that it should first be
        verified that the number is a power of 2, since
        this function will only ever return an integer
        @return: floor( log base 2 of n )
        """
        cdef int count = 0
        while n != 1:
            count += 1
            n >>= 1
        return count


    cpdef performStft(self, cnp.ndarray[dtype=cnp.float32_t, mode='c'] arr):
        """
        Perform a realtime stft
        @type data: np.ndarray
        @param data: input data for stft
        """
        cstft.performSTFT(&self._c_stft, <cnp.float32_t *> arr.data)

    cpdef performIStft(self, cnp.ndarray[dtype=cnp.float32_t, mode='c'] arr):
        """
        Perform an inverse stft
        @type data: np.ndarray
        @param data: data to hold result of istft
        """
        #cdef cnp.ndarray[dtype=cnp.float32_t] out_arr = np.zeros(SIZE, dtype=np.float32)
        cstft.performISTFT(&self._c_stft, <cnp.float32_t *> arr.data)

    cpdef getDFTs(self):
        """
        Get the dfts associated with the current state of
        this stft object in  a list
        """
        cdef cstft.DSPSplitComplex *dft = self._c_stft.dfts
        cdef int n_dfts = self._c_stft.num_dfts
        #cdef cnp.float32_t[:,:] imags = dft[0].imagp
        #cdef float *im = dft[0].imagp
        ##for i in range(n_dfts):
        ##    reals[i] = <cnp.float32_t[:(self._dft_length/2)]> dft[i].realp
        ##    imags[i] = <cnp.float32_t[:(self._dft_length/2)]> dft[i].imagp
        ##reals = np.asarray(<cnp.float32_t[:n_dfts, :(self._dft_length/2)]> dft[0].realp)
        #reals = np.asarray(<cnp.float32_t[:2, :512]> im)
        ##imags = np.asarray(<cnp.float32_t[:(self._dft_length/2)]> dft[0].imagp)

        all_reals = []
        all_imags = []
        cdef cnp.float32_t[::1] reals
        cdef cnp.float32_t[::1] imags
        for i in range(n_dfts):
            reals = <cnp.float32_t[:(self._dft_length/2)]> dft[i].realp
            imags = <cnp.float32_t[:(self._dft_length/2)]> dft[i].imagp
            all_reals.append(reals)
            all_imags.append(imags)
        return all_reals, all_imags




