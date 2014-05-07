import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pa_tools.constants as consts
from filterhemisphereplot import FilterHemispherePlot
import plottools as ptools

class ParticleHemispherePlot(FilterHemispherePlot):
  """
  Class for plotting in a 3d hemisphere
  """
  def __init__(self, n_particles, particle_color='b', n_estimates=0, n_past_estimates=1):
    FilterHemispherePlot.__init__(
        self, n_estimates=n_estimates, n_past_estimates=n_past_estimates)
    self._n_particles = n_particles
    self._particle_color = particle_color
    self._setup_particles()
    self._setup_estimates()

  def update(self, particles, weights, estimates=None):
    self._update_particle_locs(particles)
    self._update_particle_weights(weights)
    if estimates is not None:
      self._update_estimates(estimates)
    self._update_figure()

  def _setup_particles(self):
    z = np.zeros((self._n_particles,))
    self._scatter = self._ax.scatter(z, z, z, 
        facecolors=self._particle_color, edgecolors='none')
    # Store colors for modifying
    weight_color = self._scatter.get_facecolors()
    self._weight_colors = np.kron(np.ones((self._n_particles, 1)), weight_color)

  def _update_particle_locs(self, particles):
    part_norm = np.sum(particles ** 2, axis=1)
    if np.any(abs(part_norm - 1) > .02):
      print part_norm
      particles /= np.sqrt(part_norm[:, np.newaxis])
    ptools.update_3d_scatter(self._scatter, particles)

  def _update_particle_weights(self, weights):
    self._weight_colors[:, 3] = np.minimum(weights,  1)
    self._scatter.set_facecolors(self._weight_colors)
    self._scatter._sizes = weights * 1000

