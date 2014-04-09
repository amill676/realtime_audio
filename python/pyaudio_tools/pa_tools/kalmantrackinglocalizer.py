from mattools import mattools as mat
import sys

__author__ = 'Adam Miller'
from pa_tools.audiolocalizer import AudioLocalizer
import numpy as np
import math
import constants as consts
import sys
from trackinglocalizer import TrackingLocalizer

class KalmanTrackingLocalizer(TrackingLocalizer):

  def __init__(self, mic_positions, search_space, mic_forward, mic_above, 
               trans_mat, state_cov, emission_mat, emission_cov, dft_len=512, 
               sample_rate=44100, n_theta=20, n_phi=1):
    """
    Localizes source using Kalman Filter:
          x_t = A*x_{t-1} + u,  u ~ N(0,Q)
          y_t = B*x_t + v,  v ~ N(0,R)

    The state vectors x_t have 6 components: 3 coordinates and 3 velocities.
    So x = [c_1, c_2, c_3, v_1, v_2, v_3], where c_i is associated with v_i.
    Note that since x is expected to be on a source plane, c_3 and v_3 should
    be zero. However they are included for freedom and because this will make
    linear transforms among 6 dimensional spaces more easy

    :param mic_forward: length 3 vector indicating direction mic points forward
                        This should be the positive y direction from the mic's
                        point of view
    :param mic_up: length 3 vector indicating direction mic points up. This 
                    should be the positive z direction from the mics point of
                    view
    :param trans_mat: Transition matrix A (should be 6x6 matrix). 
    :param state_cov: State noise covariance matrix Q
    :param emission_mat: Emission matrix B
    :param emission_cov: Emission covariance matrix R
    """
    TrackingLocalizer.__init__(self, mic_positions, search_space, dft_len, 
                               sample_rate, n_theta, n_phi)
    self._grid_size = self._n_theta * self._n_phi
    self._process_search_space(search_space)
    self._setup_posterior_grid()
    self._setup_state_model(mic_forward, mic_above, trans_mat, state_cov, 
                            emission_mat, emission_cov)
    #self._setup_structures()

  def _process_search_space(self, search_space):
    self._search_space = search_space
    self._planes = self._search_space.get_planes()
    self._tracking_plane = self._planes[0]
  
  def get_distribution(self, rffts):
    # Predict
    state_pred = self._transition_mat.dot(self._state_estimate)
    cov_pred = \
      self._transition_mat.dot(self._estimate_cov).dot(self._transition_mat.T)
    # Update
    # Get observation
    d = self.get_distribution_real(rffts, 'gcc')
    max_ind = np.argmax(d)
    obs = self._direction_to_plane_point(self._directions[:, max_ind])
    if obs is not None:
      # perform update step of KF
      innov = obs - self._emission_mat.dot(state_pred)
      innov_cov = self._emission_mat.dot(cov_pred).dot(self._emission_mat.T) \
                    + self._emission_cov
      K = cov_pred.dot(self._emission_mat.T).dot(np.linalg.inv(innov_cov))
      # Now update estimates
      self._state_estimate = state_pred + K.dot(innov) 
      #self._state_estimate[-1] = 1
      self._estimate_cov = (np.identity(7) - K.dot(self._emission_mat)).dot(cov_pred)
      print self._state_estimate
      #self._estimate_cov[-1, -1] = .01
      #print self._estimate_cov
    return self._distribution_from_estimates(self._state_estimate, self._estimate_cov)

  def _distribution_from_estimates(self, state_est, cov_est):
    """
    Get the distribution over possible directions from the current KF estimates
    """
    pos_est = state_est[:3]
    pos_cov = cov_est[:3, :3]
    distr = np.empty((self._grid_size,))
    for i in range(self._grid_size):
      point_on_plane = self._direction_to_plane_point(self._directions[:, i])
      if point_on_plane is None:
        distr[i] = 0
        continue
      distr[i] = mat.gauss_pdf(point_on_plane, pos_est, pos_cov)
    return distr

  def _get_prediction(self):
    """
    Get predicted next state using the current posterior estimate
    """
    return self._transition_mat.T.dot(self._posterior_grid)
  

  def _setup_posterior_grid(self):
    self._posterior_grid = (1. / (self._n_phi * self._n_theta)) * \
                            np.ones((self._n_phi * self._n_theta,))

  def _setup_state_model(self, mic_forward, mic_above, trans_mat, state_cov, 
                         emission_mat, emission_cov):
    """
    Assuming we have a matrix D such that we can transform from a point in the
    state space on the plane x to a point in the state space from the mic's
    point of view z, (which is constructed in _setup_state_transform_mat()),
    we can now construct a new state space model in teh transformed space.
    This will allow us to apply the kalman filtering in the new space.

    So we have:
      x_t = Ax_{t-1} + w
      z_t = Dx_t

    This gives us:
      z_t = DAinv(D)z_{t-1} + Dw
          = Gz_{t-1} + u

    Thus we can store the new transition matrix G and the new noise covariance
    and we should be good

    """
    # Setup the transformation matrix D
    self._setup_state_transform_mat(mic_forward, mic_above)
    # Store standard model
    #self._transition_mat = trans_mat
    self._state_cov = state_cov
    #self._emission_mat = emission_mat
    self._emission_mat = np.hstack((np.identity(3), np.zeros((3,4))))
    self._emission_cov = emission_cov
    # Transform into new space
    extended_trans_mat = np.zeros((7,7))
    extended_trans_mat[:6, :6] = trans_mat
    extended_trans_mat[6, 6] = 1
    print extended_trans_mat
    self._transition_mat = \
      self._state_transform_mat.dot(extended_trans_mat).dot(
        np.linalg.inv(self._state_transform_mat)
      )
    #print self._transition_mat
    extended_cov = np.zeros((7,7))
    extended_cov[:6, :6] = state_cov
    self._transformed_state_cov = \
      self._state_transform_mat.dot(extended_cov).dot(self._state_transform_mat.T)
    homogenous_cov = .01
    self._transformed_state_cov[6, 6] = homogenous_cov  # For homogenous coords
    # Setup intial state
    initial_pos = self._mic_basis.dot(self._tracking_plane.get_offset() - self._search_space.get_mic_loc())
    self._state_estimate = np.zeros((7,))
    self._state_estimate[:3] = initial_pos
    self._state_estimate[-1] = 1
    self._estimate_cov = np.identity(7, dtype=consts.REAL_DTYPE)


  def _setup_state_transform_mat(self, mic_forward, mic_above):
    """ 
    Setup variables for necessary transformations between physical state
    space and the state space as it will be observed by the microphones.

    This equates to a linear transformation to change the coordinates 
    into those as viewed by the microphone.

    Given a location x in the coordinates of the source plane, we would like
    to find a vector z that is the same point in the coordinates of the 
    microphone. This equates to solving the following equation:

    Uz + c = Wx + o

    Where U is the matrix of basis (columns) of the coordinate system of the
    microphone, W is the matrix of basis (columns) of the coordinate system
    of the source plane, o is the location of the origin of the source plane,
    and c is the location of the origin of the microphone coordinate system,
    both in world coordinates.

    Therefore to go from x to z, we use the following:
    z = inv(U) * (Wx + o - c)
      = inv(U)*W*x + inv(U)*(o - c)

    However, if we using homogenous coordinates with x, we can incorporate this
    into one linear transformation:

    z = Dx

    This method will construct D
    """
    mic_forward = mat.check_3d_vec_normalize(mic_forward)
    mic_above = mat.check_3d_vec_normalize(mic_above)
    print mic_above
    mic_right = np.cross(mic_forward, mic_above)
    # Setup basis matrix for mic coordinate system.
    U = np.array([mic_right, mic_forward, mic_above]).T
    self._mic_basis = U
    # Get matrix for transformation
    W = self._tracking_plane.get_transform_mat()
    o = self._tracking_plane.get_offset()
    c = self._search_space.get_mic_loc()
    shift = np.linalg.solve(U, o - c) # inv(U)*(o - c)
    # Setup D
    D = np.zeros((7, 7)) # Will have 3 locations, 3 velocities, 1 shift
    trans = np.linalg.inv(U).dot(W)
    # Locaiton transforms
    D[:3, :3] = trans
    D[:3, 6] = shift
    # Velocity transforms
    D[3:6, 3:6] = trans
    # Homeogenous coordinates
    D[6, 6] = 1  
    self._state_transform_mat = D
    print self._state_transform_mat
    print np.linalg.inv(self._state_transform_mat)

  def _direction_to_plane_point(self, direction):
    world_direction = self._mic_basis.dot(direction)
    offset = self._search_space.get_mic_loc()
    point = self._tracking_plane.line_intersection(world_direction, offset)
    if point is None:
      return None
    return np.linalg.solve(self._mic_basis, point - offset)

  #def _setup_structures(self):
  #  """
  #  Setup structures for enabling computation
  #  """
  #  # Precompute the probability of transitioning from one point in the 
  #  # state space to any other point, using the given transition model
  #  # entry (i,j) is probability of transitioning from point i to j
  #  self._transition_mat = np.empty((self._grid_size, self._grid_size))
  #  gauss_p = lambda x, mu: 1. / np.sqrt((2*np.pi)**2 * np.linalg.det(self._state_cov)) * \
  #              np.exp(-.5 * (x - mu).T.dot(self._state_prec).dot(x - mu))
  #  for i in range(self._grid_size):
  #    for j in range(self._grid_size):
  #      # Get current and next state in 2-dimensional plane coordinates, 
  #      # with velocities
  #      curr_state = self._search_space.get_source_loc(self._directions[:, i])
  #      if curr_state is None: # Given direction can't reach plane
  #        self._transition_mat[i, j] = 0
  #        continue
  #      curr_state = self._tracking_plane.to_plane_coordinates(curr_state)[:-1]
  #      #curr_state = np.hstack((curr_state, np.zeros((2,))))
  #      next_state = self._search_space.get_source_loc(self._directions[:, j])
  #      if next_state is None: # Given direction can't reach plane
  #        self._transition_mat[i, j] = 0
  #        continue
  #      next_state = self._tracking_plane.to_plane_coordinates(next_state)[:-1]
  #      #next_state = np.hstack((next_state, np.zeros((2,))))
  #      # Store probabilty of transition
  #      self._transition_mat[i, j] = gauss_p(next_state, curr_state)
  
