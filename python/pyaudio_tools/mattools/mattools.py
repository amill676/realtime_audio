__author__ = 'Adam Miller'
import numpy as np
import pa_tools.constants as consts
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
    fft[1:dft_len / 2] = np.asarray(reals)[1:dft_len / 2] + 1j * np.asarray(imags[1:dft_len / 2])
    fft[-1:dft_len / 2:-1] = np.asarray(reals)[1:dft_len / 2] -1j * np.asarray(imags[1:dft_len / 2])
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
    # By using a slice instead of a for loop hear we get ~1000 times speed improvement...
    fft[1:half_dft_len -1] = np.asarray(reals, dtype=np.float32)[1:half_dft_len - 1] + \
                             1j * np.asarray(imags, dtype=np.float32)[1:half_dft_len - 1]
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

def to_all_real_matlab_format(dfts):
    """
    Converts list in format of getDFTs() output to matlab format dfts
    using only the positive frequencies and nyquist frequency. Will
    return one for each "hop" given in the input.

    Each row corresponds to an input channel, and contains a list of
    positive DFTs (with all values, complex) for the data contained in the
    entry corresponding to that channel in dfts, with each entry in the list
    corresponding to the associated hop in the input list
    
    :param dfts: list of tuples containg list of real/imag lists
    :return: np.ndarray
    """
    half_dft_len = len(dfts[0][0][0]) + 1
    num_chan = len(dfts)
    num_hops = len(dfts[0][0])
    dft_arr = np.empty((num_chan, half_dft_len, num_hops), dtype=consts.COMPLEX_DTYPE)
    for i in range(num_chan):
        for k in range(num_hops):
            reals = dfts[i][0][k]
            imags = dfts[i][1][k]
            dft_arr[i, :, k] = to_real_fft(reals, imags)
    return dft_arr

def set_dft_real(reals, imags, rfft):
    if len(reals) != len(imags):
        raise ValueError("reals and imags should have same length")
    if len(reals) + 1 != len(rfft):
        raise ValueError("len(rfft) should be 1 + len(reals)")
    reals[0] = np.real(rfft[0])
    imags[0] = np.real(rfft[-1])
    half_dft_len = len(reals)
    reals[1:half_dft_len] = np.ascontiguousarray(np.real(rfft)[1:half_dft_len], dtype=np.float32)
    imags[1:half_dft_len] = np.ascontiguousarray(np.imag(rfft)[1:half_dft_len], dtype=np.float32)

# Use for setting dfts using result from beamformer
def set_dfts_real(dfts, all_rffts, n_channels=None):
    if n_channels is None or n_channels > len(dfts):
        n_channels = len(dfts)
    n_hops = len(dfts[0][0])
    for n in range(n_channels):
        reals = dfts[n][0]
        imags = dfts[n][1]
        for k in range(n_hops):
            set_dft_real(reals[k], imags[k], all_rffts[k, :])

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

def replace_n_chans_real(n_chans, dfts, rfft):
    """
    Will replace the first n channels of the dfts object
    with the fft corresponding to rfft.
    :param n_chans: number of channels to replace
    :param dfts: object output from getDFTs() call to StftManger
    :param rfft: the positve half of an fft to use in replacement
    """
    pass

def dft_mult(reals, imags, rfft):
    if len(reals) != len(imags):
        raise ValueError("reals and imags length must be same")
    if len(reals) + 1 != len(rfft):
        raise ValueError("rfft must be of length 1 + len(reals)")
    n = len(reals)
    reals[0] *= np.real(rfft[0])
    imags[0] *= np.real(rfft[-1])  # Nyquist
    for k in range(1, n):
        new_r = reals[k] * np.real(rfft[k]) - imags[k] * np.imag(rfft[k])
        new_i = reals[k] * np.imag(rfft[k]) + imags[k] * np.real(rfft[k])
        reals[k] = new_r
        imags[k] = new_i

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


def check_3d_vec(vec):
    """
    Ensure that the input is a 3-d vector and has float type.
    :returns: The float version of the vector if it has proper size. If already
                float, no copy is made
    """
    vec = check_vec(vec)
    if len(vec) != 3:
        raise ValueError("vectors must be 3 dimensional")
    return vec

def check_3d_vec_normalize(vec):
    """
    Ensure that the input is a 3-d vector and has float type.
    :returns: The float version of the vector if it has proper size. If already
                float, no copy is made
    """
    vec = check_3d_vec(vec)
    if norm2(vec) < consts.EPS:
      ValueError("Cannot normalize length 0 vector")
    vec /= norm2(vec)
    return vec

def check_vec(vec):
    """
    Ensure that the input is a vector and has float type.
    :returns: the float version of the vector if it is a vector.
                 Raises ValueError otherwise
    """
    if len(vec.shape) != 1:
        raise ValueError("vectors must be a numpy vector array (should have only one axis)")
    return to_float(vec)


def to_float(arr):
    """
    Return the array with a floating point dtype. If the array
    already has such a dtype, it will not be copied, but simply
    returned
    :param arr: numpy array to be converted
    """
    dtype = arr.dtype
    if not (dtype is np.float16 or dtype is np.float32 or dtype is np.float64 or
                    dtype is np.complex64 or dtype is np.complex128):
        return arr.astype(np.float64)  # Same as np.float
    return arr


def norm2(vec):
    """
    Return the 2-norm of a vector
    """
    vec = check_vec(vec)
    return (float(np.sum(vec ** 2))) ** .5

def gauss_pdf(x, mu, cov):
    scaled = np.linalg.solve(cov, x-mu)
    return 1. / np.sqrt((2*np.pi)**2 * np.linalg.det(cov)) * \
                np.exp(-.5 * (x - mu).T.dot(scaled))


def plane_intersection_point(basis, source_offset, source_dir, 
                             plane_offset, plane_normal):
    """
    Find the point at which a line eminating from a source will hit a plane.
    Express the line in the coordinate system of the source
    :param basis: 3x3 matrix where columns are basis that the line direction
                  will be expressed in
    :param source_offset: offset to origin of source coordinate system, in
                          world coordinates (3-d vector)
    :param source_dir: 3-d vector expressing direction of line from source.
                       This is expressed in the source coordinates, not world
                       coordinates
    :param plane_offset: offset to point on plane (in world coordinates)
    :param plane_normal: normal vector to plane in world coordinates
    """
    vec = plane_normal.T.dot(plane_offset - source_offset) / \
          plane_normal.T.dot(basis.dot(source_dir)) * basis.dot(source_dir)
    return np.linalg.solve(basis, vec)

def add_frame(mat, frame):
  """
  Update a matrix of frames given a new frame. 
  :param mat: m x n matrix with n frames, each of m dimensions
  :param frame: new frame to put in last column of matrix. Should be vector
                of m dimensions
  """
  if len(mat.shape) > 1:
    mat[:, :-1] = mat[:, 1:]

    mat[:, -1] = frame
  else:
    mat[:-1] = mat[1:]
    mat[-1] = frame

def add_3d_frame(mat, frame):
  """
  Update a 3d matrix of frames given a new frame. 
  :param mat: m x n x p matrix with n frames, each of m x p dimensions
  :param frame: new frame to put in last column of matrix. Should be matrix
                of m x p dimensions
  """
  mat[:, :-1, :] = mat[:, 1:, :]
  mat[:, -1, :] = frame
