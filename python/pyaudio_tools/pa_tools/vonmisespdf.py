from pybayes import Pdf
from pa_tools.vonmisescpdf import VonMisesCPdf
import scipy.special as sps
import numpy as np
import constants as consts
import numpy.random
import mattools as tools


class VonMisesPdf(Pdf):
  def __init__(self, mu, kappa, rv):
    """
    :param mu: mu of von mises distribution - should be on unit sphere
    :param kappa: concentration of von mises-fisher distribution
    :param rv: associated random variable (always set in constructor, 
               contains at least one RVComp
    """
    Pdf.__init__(kappa, rv)
    self._process_inputs(mu, kappa, rv)

  def _process_inputs(self, mu, kappa, rv):
    if rv.dimension != self._ndim(mu):
      raise ValueError("RV and mu must have same number of dimensions")
    if tools.norm2(mu) < consts.EPS:
      raise ValueError("mu must have non-zero length")
    self._n_dimensions = self._ndim(mu)
    mu /= tools.norm2(mu)
    self._mu = mu
    self._cpdf = VonMisesCPdf(kappa, rv)

  def samples(self, n, cond=None):
    return self._cpdf.samples(n, self._mu)

  def shape(self):
    return self._n_dimensions

  def sample(self, cond=None):
    return self._cpdf.sample(self._mu)
 
  def mean(self, cond=None):
    return self._mu

  def eval_log(self, x, cond=None):
    return self._cpdf.eval_log(x, self._mu)

  def _ndim(self, x):
    return x.shape[0]

