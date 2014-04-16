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

class VonMisesTrackingLocalizer(TrackingLocalizer):

  def __init__(self, mic_positions, search_space, n_particles,
               state_kappa, observation_kappa, outlier_prob=0, 
               dft_len=512, sample_rate=44100, n_theta=20, n_phi=1):
    """
    Localizes source using Von Mises particle filter
          x_t ~ VM(x_{t-1}, kappa_v)
          y_t ~ VM(x_t, kappa_w)

    :param n_particles: number of particles to use
    :param state_kappa: concentration parameter for state von mises distribution
    :param observation_kappa: concentration parameter for observation von mises
                              distribution
    :param outlier_prob: Probability that a given observation comes from a 
                         background uniform outlier von mises distribution. If
                         this is omitted, it will be set to 0, and the normal
                         particle filtering algorithm will be used

    """
    TrackingLocalizer.__init__(self, mic_positions, search_space, dft_len, 
                               sample_rate, n_theta, n_phi)
    self._grid_size = self._n_theta * self._n_phi
    self._process_search_space(search_space)
    self._setup_particle_filters(n_particles, state_kappa, observation_kappa, outlier_prob)

  def _process_search_space(self, search_space):
    self._search_space = search_space
    self._planes = self._search_space.get_planes()
    self._tracking_plane = self._planes[0]

  def get_distribution(self, rffts):
    d, energy = self.get_distribution_real(rffts, 'gcc')
    maxind = np.argmax(d)
    obs = self._directions[:, maxind]
    obs = np.asarray(obs, dtype=float) # port audio uses 32, pybayes uses 64
    if self._use_outlier_distribution():
      self._weighted_bayes(obs)
    else:
      self._bayes(obs)
    #self._particle_filter.bayes(obs)
    #self._posterior = self._particle_filter.posterior()
    return self._posterior

  def _setup_particle_filters(self, n_particles, state_kappa, 
                              observation_kappa, outlier_prob):
    """
    Setup the distributions needed by PartcileFilter in pybayes
    """
    sys.stdout.flush()
    self._n_particles = n_particles
    self._state_kappa = state_kappa
    self._obs_kappa = observation_kappa
    ndim = self._get_effective_n_dimensions()
    # State RVs
    self._x_t = pb.RV(pb.RVComp(ndim, 'x_t'))
    self._x_t_1 = pb.RV(pb.RVComp(ndim, 'x_{t-1}'))
    # Observation RV
    self._y_t = pb.RV(pb.RVComp(ndim, 'y_t'))
    # Initial state RV
    self._x0 = pb.RV(pb.RVComp(ndim, 'x0'))
    init_kappa = .5 # Really small so is almost uniform
    init_mu = np.ones((ndim,))


    # Create distributions
    self._state_distribution = \
      VonMisesCPdf(self._state_kappa, self._x_t, self._x_t_1)
    self._obs_distribution = \
      VonMisesCPdf(self._obs_kappa, self._y_t, self._x_t)
    self._init_distribution = \
      VonMisesPdf(init_mu, init_kappa, self._x0)

    # Setup distribution for outliers
    if outlier_prob < 0 or outlier_prob > 1:
      raise ValueError("Outlier probability must be between 0 and 1")
    self._outlier_rv = pb.RV(pb.RVComp(ndim, 'outlier'))
    self._outlier_mu = np.hstack((np.array([0, -1.]), np.zeros((ndim-2,))))
    self._outlier_kappa = .001
    self._outlier_prob = outlier_prob # Probability of generation from background pdf
    self._outlier_distribution = \
      VonMisesPdf(self._outlier_mu, self._outlier_kappa, self._outlier_rv)

    # Do particle filtering ourselves...
    self._posterior = EmpPdf(self._init_distribution.samples(self._n_particles))
    self._count = 0

    #self._particle_filter = pb.ParticleFilter(self._n_particles, 
    #                                          self._init_distribution, 
    #                                          self._state_distribution,
    #                                          self._obs_distribution)
    #self._posterior = self._particle_filter.posterior()
                        
  def _bayes(self, yt):
    """
    Take care of particle filtering ourselves, otherwise we don't have easy access
    to the weights of particles
    """
    # resample -- do it here so that the weights will be available after one run
    # of inference.
    self._posterior.resample()
    for i in range(self._posterior.particles.shape[0]):
      # generate new ith particle:
      self._posterior.particles[i] = \
        self._state_distribution.sample(self._posterior.particles[i])
      # recompute ith weight:
      self._posterior.weights[i] *= \
        np.exp(self._obs_distribution.eval_log(yt, self._posterior.particles[i]))
    # assure that weights are normalised
    self._posterior.normalise_weights()
    return True

  def _weighted_bayes(self, yt):
    """
    Do particle filtering using a spike and slab method. That is, assume we
    have a background near-uniform distribution eating up all the outlier
    data. Then weight the particles using this assumption
    """
    self._count += 1
    if self._count % 1 == 0:
      self._posterior.resample()
    for i in range(self._posterior.particles.shape[0]):
      # generate new ith particle:
      self._posterior.particles[i] = \
        self._state_distribution.sample(self._posterior.particles[i])
      # recompute ith weight:
      # log likelihood obs came from state (not from outlier)
      state_ll = self._obs_distribution.eval_log(yt, self._posterior.particles[i]) 
      state_post = np.log(1 - self._outlier_prob) + state_ll
      outlier_ll = self._outlier_distribution.eval_log(yt)
      outlier_post = np.log(self._outlier_prob) + outlier_ll
      log_norm_const = np.log(np.exp(state_post) + np.exp(outlier_post))
      eta_state = state_post - log_norm_const
      eta_outlier = outlier_post - log_norm_const
      weight_update = (np.exp(state_ll + eta_state) + np.exp(outlier_ll + eta_outlier))
      self._posterior.weights[i] *= weight_update
      #print "state: %f, outlier: %f, update: %f" %(eta_state, eta_outlier, weight_update)
      #self._posterior.weights[i] *= \
      #  np.exp(self._obs_distribution.eval_log(yt, self._posterior.particles[i]))
    # assure that weights are normalised
    self._posterior.normalise_weights()

  def _get_effective_n_dimensions(self):
    if self._n_phi == 1:
      return 2
    return self._n_dimensions
  
  def _use_outlier_distribution(self):
    return self._outlier_prob > 0



    
