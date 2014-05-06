import numpy as np
import matplotlib.pyplot as plt
import pa_tools.constants as consts
from filterplot import FilterPlot

class ParticleFilterPlot(FilterPlot):
  """
  Class for plotting particle filter over time. y axis will be state space
  while x axis will be time
  """
  def __init__(self, n_space, n_past_samples, n_particles, n_estimates=0, particle_color='b', title=''):
    FilterPlot.__init__(self, n_space, n_past_samples, n_estimates, title)
    self._n_particles = n_particles
    self._particle_color = particle_color
    self._setup_particles()

  def _setup_particles(self):
    self._particles = np.zeros((self._n_particles, self._n_past_samples))
    self._weights = np.zeros((self._n_particles, self._n_past_samples))

    self._scatter_space = np.kron(np.ones((self._n_particles,)), np.arange(self._n_past_samples)) 
    self._scatter = plt.scatter(self._scatter_space, 
        np.reshape(self._particles, self._n_particles * self._n_past_samples), 
        edgecolors='none', facecolors=self._particle_color, s=45)

    # Setup coloring
    color = self._scatter.get_facecolors()
    self._colors = np.kron(np.ones((self._n_particles * self._n_space, 1)), color)

  def update(self, distr, particles, weights, estimates=None):
    # Update structures
    self._update_distr(distr)
    self._update_particles(particles, weights)
    if estimates is not None:
      self._update_estimates(estimates)
    # Update plot objects
    self._update_plots()
    self._update_scatter()
    self._update_figure()

  def _update_particles(self, particles, weights):
    self._add_frame(self._particles, particles)
    self._add_frame(self._weights, weights)

  def _update_scatter(self):
    # Update plot
    self._scatter.set_offsets(
            np.array([self._scatter_space, np.reshape(self._particles, self._particles.size)]).T)
    # UPdate colors
    vec_weights = np.reshape(self._weights, self._weights.size)
    self._colors[:, 3] = np.minimum(2 *vec_weights, 1)
    self._scatter.set_facecolors(self._colors)
    #self._scatter._sizes = vec_weights * 500


