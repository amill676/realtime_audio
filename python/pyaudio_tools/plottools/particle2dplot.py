import numpy as np
import matplotlib.pyplot as plt
import pa_tools.constants as consts

class Particle2DPlot(object):
  """
  Class for plotting particle filter over time. y axis will be state space
  while x axis will be time
  """
  def __init__(self, ax, n_particles, n_space, n_past_samples, particle_color='r', title=''):
    self._ax = ax
    self._n_past_samples = n_past_samples
    self._n_particles = n_particles
    self._n_space = n_space
    self._title = title
    self._particle_color = particle_color
    self._setup()

  def _setup(self):
    self._setup_structures()
    self._setup_plot()

  def _setup_plot(self):
    self._ax.get_xaxis().set_visible(False)
    self._ax.get_yaxis().set_visible(False)
    # Create actual plot
    self._im_2d = self._ax.imshow(self._lhood_mat, vmin=0, vmax=.03, cmap='bone', origin='lower',
                      extent=[0, self._n_past_samples-1, 0, np.pi], aspect='auto')
    self._scatter_space = np.kron(np.ones((self._n_particles,)), np.arange(self._n_past_samples)) 
    self._scatter = plt.scatter(self._scatter_space, 
        np.reshape(self._particles, self._n_particles * self._n_past_samples), 
        edgecolors='none', facecolors=self._particle_color, s=45)
    # Setup limits
    self._ax.set_ylim(0, np.pi)
    self._ax.set_xlim(0, self._n_past_samples-1)
    # Setup lables
    self._ax.set_title(self._title)
    self._ax.set_xlabel('time')
    self._ax.set_ylabel('DOA')

    # Setup coloring
    color = self._scatter.get_facecolors()
    self._colors = np.kron(np.ones((self._n_particles * self._n_space, 1)), color)
    print self._colors

  def _setup_structures(self):
    self._lhood_mat = np.zeros((self._n_space, self._n_past_samples))
    self._particles = np.zeros((self._n_particles, self._n_past_samples))
    self._weights = np.zeros((self._n_particles, self._n_past_samples))

  def update(self, particles, weights, distr):
    # Update particles and weights
    # Translate particles into theta space
    self._particles[:, :-1] = self._particles[:, 1:]
    self._particles[:, -1] = particles
    # Update weights
    self._weights[:, :-1] = self._weights[:, 1:]
    self._weights[:, -1] = weights
    # Update likelihoods
    self._lhood_mat[:, :-1] = self._lhood_mat[:, 1:]
    dist = distr - np.min(distr)
    dist /= (np.sum(dist) + consts.EPS)
    self._lhood_mat[:, -1] = dist

    # Update plot
    self._im_2d.set_array(self._lhood_mat)
    self._scatter.set_offsets(
            np.array([self._scatter_space, np.reshape(self._particles, self._particles.size)]).T)
    # UPdate colors
    vec_weights = np.reshape(self._weights, self._weights.size)
    self._colors[:, 3] = vec_weights
    self._scatter.set_facecolors(self._colors)
    #self._scatter._sizes = vec_weights * 500


