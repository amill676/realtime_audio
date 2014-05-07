import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pa_tools.constants as consts
from hemisphereplot import HemispherePlot
import plottools as ptools
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class ParticleHemispherePlot(HemispherePlot):
  """
  Class for plotting in a 3d hemisphere
  """
  def __init__(self, n_particles, particle_color='b'):
    HemispherePlot.__init__(self)
    self._n_particles = n_particles
    self._particle_color = particle_color
    self._setup_particles()
    self._setup_estimate()

  def update(self, particles, weights):
    self._update_particle_locs(particles)
    self._update_particle_weights(weights)
    self._update_estimate(particles, weights)
    self._update_figure()

  def _setup_particles(self):
    z = np.zeros((self._n_particles,))
    self._scatter = self._ax.scatter(z, z, z, 
        facecolors=self._particle_color, edgecolors='none')
    # Store colors for modifying
    color = self._scatter.get_facecolors()
    self._colors = np.kron(np.ones((self._n_particles, 1)), color)

  def _setup_estimate(self):
    # Current esetimate vector
    self._estimate_plot, = self._ax.plot([0, 0], [0, 0], [0, 0], 'k')

    # Previous estimate points
    self._n_previous_estimates = 50
    self._estimates = np.zeros((3, self._n_previous_estimates))
    # Setup color fading
    self._estimate_base_color = np.array([1, 0, 0, 1])
    self._estimate_colors = np.kron(np.ones((self._n_previous_estimates, 1)), 
                                    self._estimate_base_color)
    self._estimate_colors[:, 3] = np.linspace(0, 1, self._n_previous_estimates)
    # Create 3d linecollection for storing lines between past estimates
    self._estimate_lc = Line3DCollection(self._segments_from_vectors, lw=2)
    self._estimate_lc.set_color(self._estimate_colors[1:])
    self._ax.add_collection3d(self._estimate_lc)

  def _update_particle_locs(self, particles):
    part_norm = np.sum(particles ** 2, axis=1)
    if np.any(abs(part_norm - 1) > .02):
      print part_norm
      particles /= np.sqrt(part_norm[:, np.newaxis])
    ptools.update_3d_scatter(self._scatter, particles)

  def _update_particle_weights(self, weights):
    self._colors[:, 3] = np.minimum(weights,  1)
    self._scatter.set_facecolors(self._colors)
    self._scatter._sizes = weights * 1000

  def _update_estimate(self, particles, weights):
    # Update vector estimate
    estimate = weights.dot(particles)
    self._estimate_plot.set_xdata([0, estimate[0]])
    self._estimate_plot.set_ydata([0, estimate[1]])
    self._estimate_plot.set_3d_properties([0, estimate[2]])

    # Update previous estimates
    self._add_frame(self._estimates, estimate)
    #ptools.update_3d_scatter(self._estimate_scat, self._estimates.T)
    segments = self._segments_from_vectors(self._estimates)
    self._estimate_lc.set_segments(self._segments_from_vectors(self._estimates))

  def _segments_from_vectors(self, vectors):
    """
    Create array of segments from an array of different vectors. This
    will create a block matrix that can be used as the argument to a 
    LineCOllection or Line3DCollection object. 
    Thus it will be of size N x 2 x d, where N is the number of vector
    points and d is the number of dimensions in the vector
    :param vectors: d x N array where each column is a vector point
    """
    transposed = vectors.transpose().reshape(-1, 1, 3)
    return np.hstack((transposed[:-1], transposed[1:]))

