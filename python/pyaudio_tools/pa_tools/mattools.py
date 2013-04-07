__author__ = 'Adam Miller'
import numpy as np
import constants as consts
import math


def to_numpy_format(dfts):
    """
    @rtype: numpy.ndarray
    """
    chan_num = len(dfts)
    dft_len_over_2 = len(dfts[0][0][0])  # Length of output of vDSP ifft
    new_dfts = np.empty((dft_len_over_2 * 2, chan_num), dtype=consts.REAL_DTYPE)
    for n in range(chan_num):
        (reals, imags) = dfts[n]
        # Remember that reals and imags is list - take only first dft
        zipped = zip_fft(reals[0], imags[0])
        new_dfts[:, n] = zipped
    return new_dfts


def to_full_ffts(dfts):
    """
    @rtype: numpy.ndarray
    """
    chan_num = len(dfts)
    dft_len = 2 * len(dfts[0][0][0])
    new_dfts = np.empty((dft_len, chan_num), dtype=consts.REAL_DTYPE)
    for n in range(chan_num):
        (reals, imags) = dfts[n]
        # Remember that reals and imags is list - take only first dft
        zipped = zip_fft(reals[0], imags[0])
        new_dfts[n, :] = zipped


def to_full_fft(reals, imags):
    """
    Converts a list of reals and imags in the format corresponding to
    the output of getDFTs() for StftManager into an ndarray containing
    the corresponding DFT
    :param reals: list of reals. Note that only half the spectrum should
                    be contained. Real signals are assumed
    :param imags: list of imaginary values. Corresponding to half the
                    frequencies of the DFT
    :return: numpy.ndarray
    """
    if len(reals) != len(imags):
        raise ValueError("real and imag arrays must be of same length")
    dft_len = 2 * len(reals)
    fft = np.empty(dft_len, dtype=consts.COMPLEX_DTYPE)
    fft[0] = reals[0]  # DC component stored here. Must be real - real signal
    fft[dft_len / 2] = imags[0]  # Nyquist component here. Must be real - real signal
    for i in range(1, dft_len / 2):
        fft[i] = reals[i] + 1j * imags[i]
        fft[-1 * i] = reals[i] -1j * imags[i]
    return fft


def to_real_fft(reals, imags):
    """
    Converts a list of reals and imags in the format corresponding to
    the output of getDFTs() for StftManager into an ndarray containing
    the coefficients corresponding to the positive frequencies in the
    represented DFT
    """
    if len(reals) != len(imags):
        raise ValueError("real and imag arrays must be of same length")
    half_dft_len = len(reals) + 1
    fft = np.empty(half_dft_len, dtype=consts.COMPLEX_DTYPE)
    fft[0] = reals[0]  # DC component stored here. Must be real - real signal
    fft[half_dft_len - 1] = imags[0]  # Nyquist component here. Must be real - real signal
    for i in range(1, half_dft_len - 1):
        fft[i] = reals[i] + 1j * imags[i]
    return fft


def to_matlab_format(dfts):
    """
    Converts list in format of getDFTs() output to matlab format
    Each row corresponds to an input channel, and contains a
    full DFT (with all values, complex) for the data contained in the
    entry corresponding to that channel in dfts
    :param dfts: list of tuples containg list of real/imag lists
    :return: np.ndarray
    """
    dft_len = 2 * len(dfts[0][0][0])
    num_chan = len(dfts)
    dft_arr = np.empty((num_chan, dft_len), dtype=consts.COMPLEX_DTYPE)
    for i in range(dft_arr.shape[0]):
        reals = dfts[i][0][0]
        imags = dfts[i][1][0]
        dft_arr[i, :] = to_full_fft(reals, imags)
    return dft_arr


def to_real_matlab_format(dfts):
    half_dft_len = len(dfts[0][0][0]) + 1
    num_chan = len(dfts)
    dft_arr = np.empty((num_chan, half_dft_len), dtype=consts.COMPLEX_DTYPE)
    for i in range(dft_arr.shape[0]):
        reals = dfts[i][0][0]
        imags = dfts[i][1][0]
        dft_arr[i, :] = to_real_fft(reals, imags)
    return dft_arr


def zip_fft(reals, imags):
    """
    """
    zipped = np.empty(2 * len(reals))
    zipped[0] = reals[0]  # DC component
    zipped[-1] = imags[0]  # Nyquist
    for i in range(1, len(reals)):
        zipped[2 * i - 1] = reals[i]
        zipped[2 * i] = imags[i]
    return zipped


def normalize_rows(a):
    """
    Normalize the rows of a matrix so they sum to one
    """
    for row in a:
        row_sum = np.sum(np.abs(row))
        if row_sum > 0:
            row /= row_sum
    return a


def log_normalize_rows(a):
    """
    Normalize the rows of a matrix so they sum to one
    """
    for row in a:
        row_sum = np.sum(np.abs(row))
        if row_sum > 0:
            row = np.log(row) - math.log(row_sum)
        else:
            row = np.log(row + consts.EPS)
    return a


def cholesky_solve(a, b):
    """
    Solves the normal equation a.T.dot(a).dot(x) = a.T.dot(b) for
    x using a cholesky factorization of a.T.dot(a)
    @param: a linear weight matrix
    @param: b right hand side vector
    @return: the solution x
    """
    L = np.linalg.cholesky(a.T.dot(a))
    y = np.linalg.solve(L, a.T.dot(b))
    x = np.linalg.solve(L.T, y)
    return x
