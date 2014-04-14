from pybayes import CPdf
import scipy.special as sps
import numpy as np
import constants as consts
import numpy.random
import mattools as tools

class VonMisesCPdf(CPdf):

  def __init__(self, rv, cond_rv, kappa):
    """
    :param rv: associated random variable (always set in constructor, 
               contains at least one RVComp
    :param cond_rv: associated condition random variable (set in constructor
                    to potentially empty RV). Will be used as the mean of
                    distribution
    :param kappa: concentration of von mises-fisher distribution
    """
    CPdf.__init__(rv, cond_rv)
    self._process_inputs(rv, cond_rv, kappa)


  def _process_inputs(self, rv, cond_rv, kappa):
    """
    Process inputs arguments to this class. Instantiate necessary member
    variables
    """
    if rv.dimension != cond_rv.dimension:
      raise ValueError("RV and Cond_RV must have same shape")
    self._kappa = kappa
    self._n_dimensions = rv.dimension
    if self._n_dimensions != 2 and self._n_dimensions != 3:
      raise ValueError("Only support 2 and 3 dimensional von mises distributions")

  def mean(self, cond=None):
    """
    Return (conditional) mean value of the pdf
    """
    if cond is None:
      raise ValueError("Must provide condition for mean")
    if len(cond) != self.rv.shape():
      raise ValueError("Conditional must have the same shape as the " + \
                       "conditional random variable provided")
    return cond 

  def eval_log(self, x, cond=None):
    if cond is None:
      raise ValueError("Must provide condition for eval_log")
    cond = self._verify_shape(cond)
    x = self._verify_shape(x)
    if self._ndims(cond) == 3:
      return np.log(self._kappa) - np.log(2*np.pi)  - \
          np.log(1-np.exp(-2 * self._kappa)) -self._kappa + self._kappa * cond.dot(x)
    elif self._ndims(cond) == 2:
      return -np.log(2 * np.pi * scipy.sps.jn(0, self._kappa)) + \
          self._kappa * np.cos(np.arctan2(x[1], x[0]) - np.arctan2(cond[1], cond[0]))
    else:
      return ValueError("Dimensions other than 2 or 3 not supported for von mises")


  def sample(self, cond=None):
    return self.samples(1, cond)

  def samples(self, n, cond=None):
    cond = self._verify_shape(cond)
    # Normalize -- must lie on unit sphere
    if tools.norm2(cond) < consts.EPS:
      raise ValueError("Cannot give lenth 0 vector")
    cond /= tools.norm2(cond)
    # Use correct sampling method
    if self._ndims(cond) == 2:
      return self._sample_2d(cond, n)
    elif self._ndims(cond) == 3:
      return self._sample_3d(cond, n)
    else:
      return ValueError("Dimensions other than 2 or 3 not supported for von mises")

  def _verify_shape(self, x):
    if x is None:
      raise ValueError("input cannot be None")
    if (len(x.shape) != 1) or (x.shape[0] != self._n_dimensions):
      raise ValueError("Shape is not consistent with specified dimensions")
    if not x.dtype is float:
      x = np.asarray(x, dtype=float)
    return x

  def _ndims(self, x):
    return x.shape[0]

  def _sample_2d(self, mu, n):
    """
    Assumes mu has already been ensured to be of proper size
    """
    nu = np.arctan2(mu[1], mu[0])
    theta = np.random.vonmises(nu, self._kappa, n)
    return np.array([np.cos(theta), np.sin(theta)])

  def _sample_3d(self, mu, n):
    """
    Assumes mu has already been ensured to be of proper size
    """
    u = np.random.rand(n)
    #W = (1 / self._kappa) * \
    #    np.log(np.exp(-self._kappa + 2 * np.sinh(self._kappa) * u))
    W = 1. + (1. / self._kappa) * np.log(np.exp(-2 * self._kappa) * (1. - u) + u)
    theta = np.random.rand(n) * 2 * np.pi
    V = np.array([np.cos(theta), np.sin(theta)]) # 2 x n matrix
    x = np.vstack((W, np.sqrt(1 - W**2)*V))
    # Get rotation matrix
    # Get angle between [1, 0, 0] and mu
    e = np.array([1., 0., 0.])
    print mu
    ang = np.arccos(e.dot(mu) / (tools.norm2(mu)))
    print ang
    # Get rotation axis r
    r = np.cross(e, mu)
    outer = np.outer(r, r)
    R = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]]) \
        * np.sin(ang) + (np.identity(3) - outer)*np.cos(ang) + outer
    return R.dot(x)
