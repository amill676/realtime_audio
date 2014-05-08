import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pa_tools.constants as consts
from filterhemisphereplot import FilterHemispherePlot
import plottools as ptools

class ParticleHemispherePlot(FilterHemispherePlot):
  """
  Class for plotting a particle filter on a 3d hemisphere. This
  class sublcasses FilterHemispherePlot, so it also enjoys all the abilities
  of plotting estimates as described in that class.
  """

  def __init__(self, n_particles, particle_color='b', **kwargs):
    """
    :param n_particles: Number of particles to display. This will also dictate
        how many particles are expected when providing new particles to update
        the plot
    :param particle_color: color of particle for plotting. Defaults to blue.

    Other keyword arguments are passed to FilterHemispherePlot
    """
    FilterHemispherePlot.__init__(self, **kwargs)
    self._n_particles = n_particles
    self._particle_color = particle_color
    self._setup_particles()
    self._setup_estimates()

  def update(self, particles, weights, *args, **kwargs):
    """
    Update the plot given a new set of particles, weights, 
    and possibly estimates

    :param particles: new particle locations. Should be of 
        size (n_particles x 3)
    :param weights: new particle weights. Should be numpy vector of 
        length 'n_particles'

    All other arguments arguments are passed to FilterHemispherePlot.update()
    """
    self._update_particle_locs(particles)
    self._update_particle_weights(weights)
    FilterHemispherePlot.update(self, *args, **kwargs)

  def _setup_particles(self):
    """
    Setup the structures necessary for properly plotting the particles
    """
    z = np.zeros((self._n_particles,))
    self._scatter = self._ax.scatter(z, z, z, 
        facecolors=self._particle_color, edgecolors='none')
    # Store colors for modifying
    weight_color = self._scatter.get_facecolors()
    self._weight_colors = np.kron(np.ones((self._n_particles, 1)), weight_color)

  def _update_particle_locs(self, particles):
    """
    Update any strucutres maintaing the particle locations and update
    the plot so the new set of particles are displayed
    """
    part_norm = np.sum(particles ** 2, axis=1)
    if np.any(abs(part_norm - 1) > .02):
      print part_norm
      particles /= np.sqrt(part_norm[:, np.newaxis])
    ptools.update_3d_scatter(self._scatter, particles)

  def _update_particle_weights(self, weights):
    """
    Use the new weights to properly update the display of teh particles to
    reflect the new weights
    """
    self._weight_colors[:, 3] = np.minimum(weights,  1)
    self._scatter.set_facecolors(self._weight_colors)
    self._scatter._sizes = weights * 2000

