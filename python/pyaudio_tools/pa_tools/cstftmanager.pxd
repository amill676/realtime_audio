cdef extern from "/Users/adamjmiller/Programming/audio/include/realtimestft.h":
    # Define types from realtimestft.h
    ctypedef struct DSPSplitComplex:
        float * realp
        float * imagp

    ctypedef struct realtimeSTFT:
        int def_log2n
        int num_channels

        int use_window_fcn
        int window_len
        int hop_size
        float * window_buf

        float * in_buf
        int curr_in_ind

        float *out_buf
        int curr_out_ind

        int num_dfts
        DSPSplitComplex *dfts

    ctypedef enum stft_error:
        STFT_OK = 0,
        STFT_FAILED_MALLOC = 1,
        STFT_INVALID_HOPSIZE = 2,
        STFT_INVALID_WINDOWSIZE = 3,
        STFT_INVALID_DFTLEN = 4,
        STFT_INVALID_NUM_CHANNELS = 5,
        STFT_NULL_PARAMETER = 6,
        STFT_FFTSETUP_ERROR = 7,
        STFT_INVALID_DATA_SIZE = 8

    # Declare methods from realtimestft.h
    stft_error createRealtimeSTFT( realtimeSTFT *,
                                   int dft_logn,
                                   int window_logn,
                                   int hop_logn,
                                   int n_channels,
                                   int use_window_fcn,
                                   int data_size )
    stft_error destroyRealtimeSTFT( realtimeSTFT * )
    stft_error performSTFT( realtimeSTFT *, float * )
    stft_error performISTFT( realtimeSTFT *, float * )






