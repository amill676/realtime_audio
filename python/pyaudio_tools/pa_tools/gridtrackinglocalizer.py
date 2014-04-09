from mattools import mattools as mat
import sys

__author__ = 'Adam Miller'
from pa_tools.audiolocalizer import AudioLocalizer
import numpy as np
import math
import constants as consts
import sys
from trackinglocalizer import TrackingLocalizer

class GridTrackingLocalizer(TrackingLocalizer):

  def __init__(self, mic_positions, search_space, source_cov, dft_len=512, sample_rate=44100,
                     n_theta=20, n_phi=1):
    """


    """
    TrackingLocalizer.__init__(self, mic_positions, search_space, dft_len, 
                               sample_rate, n_theta, n_phi)
    self._grid_size = self._n_theta * self._n_phi
    self._process_search_space(search_space)
    self._setup_posterior_grid()
    self._setup_state_model(source_cov)
    self._setup_structures()

  def _process_search_space(self, search_space):
    self._search_space = search_space
    self._planes = self._search_space.get_planes()
    self._tracking_plane = self._planes[0]
  
  def get_distribution(self, rffts):
    d = self.get_distribution_real(rffts, 'gcc')
    #d = np.arange(self._grid_size, dtype=consts.REAL_DTYPE)
    #d = np.hstack((100 * np.ones((5,)), np.ones((self._grid_size-5,))))
    print "before: " + str(self._posterior_grid)
    pred = self._get_prediction()
    #pred /= (np.sum(pred) + consts.EPS)
    d /= (np.sum(d) + consts.EPS)
    self._posterior_grid = d * pred
    self._posterior_grid /= np.sum(self._posterior_grid)
    print "after: " + str(self._posterior_grid)
    return self._posterior_grid
    #return pred


  def _get_prediction(self):
    """
    Get predicted next state using the current posterior estimate
    """
    return self._transition_mat.T.dot(self._posterior_grid)
  

  def _setup_posterior_grid(self):
    self._posterior_grid = (1. / (self._n_phi * self._n_theta)) * \
                            np.ones((self._n_phi * self._n_theta,))

  def _setup_state_model(self, source_cov):
    # Transition parameters in state space (in physical space)
    self._transition_mat = np.identity(2) # theta, phi, dtheta, dphi
    if source_cov.shape != (2, 2):
      raise ValueError("Covariance of source location should be a 2x2 matrix")
    self._state_cov = source_cov
    self._state_prec = np.linalg.inv(self._state_cov)

  def _setup_state_transform(self):
    """ 
    Setup variables for necessary transformations between physical state
    space and the state space as it will be observed by the microphones.

    This equates to a linear transformation to change the coordinates 
    into those as viewed by the microphone
    """

  def _setup_structures(self):
    """
    Setup structures for enabling computation
    """
    # Precompute the probability of transitioning from one point in the 
    # state space to any other point, using the given transition model
    # entry (i,j) is probability of transitioning from point i to j
    self._transition_mat = np.empty((self._grid_size, self._grid_size))
    gauss_p = lambda x, mu: 1. / np.sqrt((2*np.pi)**2 * np.linalg.det(self._state_cov)) * \
                np.exp(-.5 * (x - mu).T.dot(self._state_prec).dot(x - mu))
    for i in range(self._grid_size):
      for j in range(self._grid_size):
        # Get current and next state in 2-dimensional plane coordinates, 
        # with velocities
        curr_state = self._search_space.get_source_loc(self._directions[:, i])
        if curr_state is None: # Given direction can't reach plane
          self._transition_mat[i, j] = 0
          continue
        curr_state = self._tracking_plane.to_plane_coordinates(curr_state)[:-1]
        #curr_state = np.hstack((curr_state, np.zeros((2,))))
        next_state = self._search_space.get_source_loc(self._directions[:, j])
        if next_state is None: # Given direction can't reach plane
          self._transition_mat[i, j] = 0
          continue
        next_state = self._tracking_plane.to_plane_coordinates(next_state)[:-1]
        #next_state = np.hstack((next_state, np.zeros((2,))))
        # Store probabilty of transition
        self._transition_mat[i, j] = gauss_p(next_state, curr_state)
