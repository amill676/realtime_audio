import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pa_tools.constants as consts
from hemisphereplot import HemispherePlot
import itertools
import mattools.mattools as mtools

class FilterHemispherePlot(HemispherePlot):
  """
  Class for plotting filter estimates on a 3d hemisphere.
  Allows users to provide estimates and track the estimates on the
  sphere over time
  """

  def __init__(self, n_estimates=0, n_past_estimates=1):
    HemispherePlot.__init__(self)
    self._n_estimates = n_estimates
    self._n_past_estimates = n_past_estimates
    self._estimate_colors_strings = ['r', 'b', 'g', 'k']

  def _setup_estimates(self):
    if self._n_estimates > 0:
      if self._n_past_estimates < 1:
        raise ValueError("Number of past estimates to keep must be at least 1")
      self._setup_estimate_colors()
      self._setup_estimate_vecs()
      self._setup_past_estimates()
    
  def _setup_past_estimates(self):
      # Previous estimate points
      self._estimates = np.zeros((3, self._n_past_estimates, self._n_estimates))
      # Setup color fading
      # Create 3d linecollection for storing lines between past estimates
      self._estimate_lcs = []
      for i in range(self._n_estimates):
        lc = Line3DCollection(self._segments_from_vectors, lw=2)
        lc.set_color(self._estimate_colors[:-1, :, i])
        self._ax.add_collection3d(lc)
        self._estimate_lcs.append(lc)

  def _setup_estimate_vecs(self):
    self._estimate_plots = []
    colors_cycle = itertools.cycle(self._estimate_colors_strings)
    for i in range(self._n_estimates):
      self._estimate_plots.append(
          self._ax.plot([0, 0], [0, 0], [0, 0], next(colors_cycle), lw=.5)[0]
      )

  def _setup_estimate_colors(self):
    colors_cycle = itertools.cycle(self._estimate_colors_strings)
    # First set up list of matplotlib colors to be used in plotting arguments
    self._colors = np.empty((self._n_estimates, 4))
    for i in range(self._n_estimates):
      self._colors[i, :] = colorConverter.to_rgba(next(colors_cycle))
    # Now set up arrays to be used to make the past estimates fade out after time
    # Should be size (n_past_estimates x 4 x n_estimates)
    # each entry along the third dimension corresponds to a different estimate, 
    # while each entry along the first dimension corresponds to a given past estimate
    # each entry along the second dimension corresponds to r, g, b, or alpha
    self._estimate_colors = \
      np.kron(np.ones((self._n_past_estimates, 1, 1)), 
          self._colors[:, :, np.newaxis].transpose((2, 1, 0)))
    self._estimate_colors[:, 3, :] = \
        np.kron(np.ones((1, self._n_estimates)), 
          np.linspace(0, 1, self._n_past_estimates)[:, np.newaxis])

  def _update_estimates(self, estimates):
    if len(estimates) != self._n_estimates:
      raise ValueError("Number of estimates supplied does not equal number" + \
          "of estimates supplied at instantiation")
    # Update vector estimate and form matrix of estimates
    for i, (plot, estimate) in enumerate(zip(self._estimate_plots, estimates)):
      plot.set_xdata([0, estimate[0]])
      plot.set_ydata([0, estimate[1]])
      plot.set_3d_properties([0, estimate[2]])
      # Update previous estimates
      self._add_frame(self._estimates[:, :, i], estimate)
      segments = self._segments_from_vectors(self._estimates[:, :, i])
      self._estimate_lcs[i].set_segments(segments)

  def update(self, estimates=None):
    if estimates is not None:
      self._update_estimates(estimates)
    self._update_figure

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

  
