from mattools import mattools as mat
import sys

__author__ = 'Adam Miller'
import numpy as np
import pybayes as pb
import math
import constants as consts
import sys
from trackinglocalizer import TrackingLocalizer
from pa_tools.vonmisescpdf import VonMisesCPdf
from pa_tools.vonmisespdf import VonMisesPdf
from pybayes import EmpPdf

class SRPPFTrackingLocalizer(TrackingLocalizer):

  def __init__(self, n_particles, state_kappa, *args, **kwargs):
    """
    Localizes source using Von Mises particle filter
          x_t ~ VM(x_{t-1}, kappa_v)
          y_t ~ SRPLikelihood(x_t)

    :param n_particles: number of particles to use
    :param state_kappa: concentration parameter for state von mises distribution
    
    All other parameters will be passed to TrackingLocalizer in the form of *args
    and **kwargs
    """
    TrackingLocalizer.__init__(self, *args, **kwargs)
    self._grid_size = self._n_theta * self._n_phi
    self._setup_particle_filters(n_particles, state_kappa)

  def get_distribution(self, rffts):
    self._doa_bayes(rffts)
    return self._posterior

  def _setup_particle_filters(self, n_particles, state_kappa):
    """
    Setup the distributions needed by PartcileFilter in pybayes
    """
    sys.stdout.flush()
    self._n_particles = n_particles
    self._state_kappa = state_kappa
    ndim = self._get_effective_n_dimensions()
    # State RVs
    self._x_t = pb.RV(pb.RVComp(ndim, 'x_t'))
    self._x_t_1 = pb.RV(pb.RVComp(ndim, 'x_{t-1}'))
    # Initial state RV
    self._x0 = pb.RV(pb.RVComp(ndim, 'x0'))
    init_kappa = .5 # Really small so is almost uniform
    init_mu = np.ones((ndim,))

    # Create distributions
    self._state_distribution = \
      VonMisesCPdf(self._state_kappa, self._x_t, self._x_t_1)
    self._init_distribution = \
      VonMisesPdf(init_mu, init_kappa, self._x0)

    # Do particle filtering ourselves...
    self._posterior = EmpPdf(self._init_distribution.samples(self._n_particles))
    self._estimate = self._get_estimate()
    self._count = 0

  def _doa_bayes(self, rffts):
    """
    Particle filtering using SRP-PHAT as likelihood measure of observation
    """
    # resample -- do it here so that the weights will be available after one run
    # of inference.
    self._posterior.resample()
    for i in range(self._posterior.particles.shape[0]):
      # generate new ith particle:
      self._posterior.particles[i] = \
        self._state_distribution.sample(self._posterior.particles[i])
    # Get SRP likelihood
    particles_3d = self._to_3d_particles(self._posterior.particles).T
    # Get likelihoods
    srp = self._get_srp_likelihood(rffts, particles_3d)
    srp -= np.min(srp)
    srp /= (np.sum(srp) + consts.EPS)
    self._posterior.weights *= srp
    # assure that weights are normalised
    self._posterior.normalise_weights()
    return True

  def _get_effective_n_dimensions(self):
    if self._n_phi == 1:
      return 2
    return self._n_dimensions

  def _to_3d_particles(self, mat):
    """
    Change matrix so that instead of each column being in 2d, each column is
    in 3d. This equates to adding a zero to each column vector
    """
    if self._get_effective_n_dimensions() == 3:
      return mat
    return np.hstack((np.asarray(mat), np.zeros((mat.shape[0], 1))))

  def _get_estimate(self):
      w = np.asarray(self._posterior.weights)
      parts = np.asarray(self._posterior.particles)
      return w.dot(parts)

