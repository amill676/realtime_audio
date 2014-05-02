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

  def __init__(self, n_particles, state_kappa, observation_kappa, outlier_prob=0, 
               *args, **kwargs):
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
    
    All other parameters will be passed to TrackingLocalizer in the form of *args
    and **kwargs
    """
    TrackingLocalizer.__init__(self, *args, **kwargs)
    self._grid_size = self._n_theta * self._n_phi
    #self._process_search_space(search_space)
    self._setup_particle_filters(n_particles, state_kappa, observation_kappa, outlier_prob)

  #def _process_search_space(self, search_space):
  #  self._search_space = search_space
  #  self._planes = self._search_space.get_planes()
  #  self._tracking_plane = self._planes[0]

  def get_distribution(self, rffts):
    #d, energy = self.get_distribution_real(rffts, 'gcc')
    #maxind = np.argmax(d)
    #obs = self._directions[:, maxind]
    #obs = np.asarray(obs, dtype=float) # port audio uses 32, pybayes uses 64
    #self._bayes(obs)
    #if self._use_outlier_distribution():
    #  self._weighted_bayes(obs)
    #else:
    self._doa_bayes(rffts)
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
    self._estimate = self._get_estimate()
    self._count = 0
    # Create a set of weights for tracking the distribution p(c_t|x_t,y_{1:t})
    self._class_weights = np.array([.5, .5])
    # Matrix used to store weights for each particle for each class in order
    # to calculate class posterior weights and state posterior weights
    self._joint_weights = np.ones((2, self._n_particles,)) 

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
    srp = self._get_srp_likelihood(rffts, particles_3d)
    srp -= np.min(srp)
    srp /= (np.sum(srp) + consts.EPS)
    self._posterior.weights *= srp
    #for i in range(self._posterior.particles.shape[0]):
    #  # recompute ith weight:
    #  self._posterior.weights[i] *= \
    #    np.exp(self._obs_distribution.eval_log(yt, self._posterior.particles[i]))
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
      self._weighted_resample()
      #self._posterior.resample()
    class_weight_sum = np.zeros((2,))
    for i in range(self._posterior.particles.shape[0]):
      # generate new ith particle:
      self._posterior.particles[i] = \
        self._state_distribution.sample(self._posterior.particles[i])
      # recompute ith weight:
      # Get likelihoods of classes p(y_t|c_t=j,x_t) for each class c_j
      state_ll = self._obs_distribution.eval_log(yt, self._posterior.particles[i]) 
      outlier_ll = self._outlier_distribution.eval_log(yt)
      # calculate p(c_t=j|x_t,y_{1:t-1})
      state_class_prob = np.log(1. - self._outlier_prob) + np.log(self._class_weights[0])
      outlier_class_prob = np.log(self._outlier_prob) + np.log(self._class_weights[1])
      # Class weights
      self._joint_weights[0, i] = np.exp(state_ll + state_class_prob)
      self._joint_weights[1, i] = np.exp(outlier_ll + outlier_class_prob)
      class_weight_sum += self._joint_weights[:, i]
      self._posterior.weights[i] *= np.sum(self._joint_weights[:, i])
      # Now find p(c_j|x_t) = p(x_t|c_j)p(c_j) / p(x_t) for each class c_j
      # First compute p(x_t, c_j) = p(x_t|c_j)p(c_j)
      #state_particle_log_prob = \
      #  self._obs_distribution.eval_log(self._posterior.particles[i], \
      #          self._estimate) + np.log(1 - self._outlier_prob)
      #outlier_particle_log_prob = \
      #  self._outlir_distribution.eval_log(self._posterior.particles[i]) \
      #          + np.log(self._outlier_prob)
      ## Now compute p(x_t) = \sum_j p(x_t, c_j)
      #total_particle_log_prob = np.log(np.exp(state_particle_log_prob) + \
      #                          np.exp(outlier_particle_log_prob))
      ## Finally compute the actual posteriors p(c_j|x_t)
      #state_class_post = state_particle_log_prob - total_particle_log_prob
      #outlier_class_post = outlier_particle_log_prob - total_particle_log_prob
      ## Now compute total likelihood 
      ## p(y_t|x_t) = sum_j p(y_t,c_j|x_t) = sum_j p(y_t|c_j,x_t)p(c_j|x_t)
      ##total_likelihood = np.exp(state_ll + state_class_post) + \
      ##                  np.exp(outlier_ll + outlier_class_post)
      #total_likelihood = state_ll + np.log(1 - self._outlier_prob) + \
      #                  outlier_ll + np.log(self._outlier_prob)
      ## We want to calculate p(x_t,c_state|y_t) = p(x_t|c_0,y_t)p(c_0|y_t)
      ## we have p(x_t|c_0,y_t) from above. Now calculate p(c_0|y_t)
      ## p(c_0|y_t) \propto p(y_t|c_0)p(c_0)
      #state_data_joint = self._obs_distribution.eval_log(yt, self._estimate) \
      #    + np.log(1 - self._outlier_prob)
      #outlier_data_joint = self._outlier_distribution.eval_log(yt) + \
      #    np.log(self._outlier_prob)
      #total_data_lhood = np.log(np.exp(state_data_joint) + np.exp(outlier_data_joint))
      #state_class_data_post = state_data_joint - total_data_lhood
      #outlier_class_data_post = outlier_data_joint - total_data_lhood
      #self._posterior.weights[i] *= np.exp(state_ll + total_data_lhood)
      #print "state: %f, outlier: %f, update: %f" %(eta_state, eta_outlier, weight_update)
    # assure that weights are normalised
    total_sum = np.sum(class_weight_sum)
    self._joint_weights /= (total_sum + consts.EPS)
    self._class_weights = class_weight_sum / (total_sum + consts.EPS)
    self._posterior.normalise_weights()
    self._estimate = self._get_estimate()

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
    return np.hstack((np.asarray(mat), np.zeros((1, mat.shape[0]))))

  def _get_estimate(self):
      w = np.asarray(self._posterior.weights)
      parts = np.asarray(self._posterior.particles)
      return w.dot(parts)

  def get_joint_weights(self):
    return self._joint_weights.copy()

  def get_class_weights(self):
    return self._class_weights
  
  def _use_outlier_distribution(self):
    return self._outlier_prob > 0

  def _weighted_resample(self):
    resample_idxs = self._posterior.get_resample_indices()
    # Resample particles and associated joint weights
    np.asarray(self._posterior.particles)[:] = np.asarray(self._posterior.particles)[resample_idxs]
    self._joint_weights[:, :] = self._joint_weights[:, resample_idxs]
    # Normalize joint weights -- ensures particle weights normalized
    self._joint_weights /= (np.sum(self._joint_weights, axis=0) * self._n_particles)
    self._posterior.weights[:] = 1. / self._n_particles
    self._class_weights = np.sum(self._joint_weights, axis=1)
    self._class_weights = np.array([.5, .5])



    
