import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors
from matplotlib.colors import colorConverter
import pa_tools.constants as consts
from hemisphereplot import HemispherePlot
import plottools as ptools
import itertools
import mattools.mattools as mtools

class FilterHemispherePlot(HemispherePlot):
  """
  Class for plotting filter estimates on a 3d hemisphere.
  Allows users to provide estimates and track the estimates on the
  sphere over time. 
  """
  
  def __init__(self, n_estimates=0, n_past_estimates=1, 
               plot_lines=None, estim_colors=None, **kwargs):
    """
    Have ability to display previous estimates, and how many to keep. Regardless
    of the amount, previous plotted estimates will fade away over time.
    Also have the option to draw lines between sequential. If not, scatter 
    points are drawn. Either way, previous estimates will still fade over time.

    :param n_estimates: number of estimates to track. Defaults to 0, in which
                        case no estimates are displayed
    :param n_past_estimates: number of previous estimates to display in plot. As
        noted above, these will fade away after time to avoid clutter in the 
        plot
    :param plot_lines: Controls whether to plot a trail bewteen all previous 
        estimates. Should be in the format of a list that is 'n_estimates'
        long. Each element should be a boolean value that is True if the
        associated previous estimates should have a trail plotted between
        them. Otherwise scatter points will be plotted at the points of all 
        previous and current values of the corresponding estimate.

    Other keyword arguments are passed to HemispherePlot
    """
    HemispherePlot.__init__(self, **kwargs)
    self._n_estimates = n_estimates
    self._n_past_estimates = n_past_estimates
    if estim_colors is not None:
      self._estimate_colors_strings = estim_colors
    else:
      self._estimate_colors_strings = ['r', 'b', 'g', 'k']
    self._setup_plot_settings(plot_lines)

  def _setup_plot_settings(self, plot_lines):
    """
    Setup the structures used to keep track of which estimates will have a line
    plot between previous estimates and which will simply use scatter points. 
    The scatter option helps prevent the plot from getting very messy when the
    estimates vary a lot.
    """
    if plot_lines is None:
      self._plot_lines = [True] * self._n_estimates
      self._plot_scatters = [False] * self._n_estimates
      self._do_plot_lines = True
      self._do_plot_scatters = False
    elif len(plot_lines) != self._n_estimates:
      raise ValueError("Number of values in plot_lines array does not equal" +\
            " the number of estimates supplied")
    else:
      # Check if we have any lines or all lines for optimizing checks later on
      any_lines = False
      all_lines = True
      for val in plot_lines:
        if type(val) is not bool:
          raise ValueError("Values in plot_lines list should be of type bool")
        any_lines = (any_lines or val)
        all_lines = (all_lines and val)
      self._plot_lines = plot_lines
      self._plot_scatters = map(lambda x: not x, self._plot_lines)
      self._do_plot_lines = any_lines
      self._do_plot_scatters = not all_lines

  def _setup_estimates(self):
    """
    Setup the structure to hold all previous values of the different estimates.
    Also call the appropriate functions to setup the different structures 
    needed to display those values
    """

    if self._n_estimates > 0:
      if self._n_past_estimates < 1:
        raise ValueError("Number of past estimates to keep must be at least 1")
      # Setup matrix for storing all past estimates
      self._estimates = np.zeros((3, self._n_past_estimates, self._n_estimates))
      # Setup all structures and settings for plotting estimates
      self._setup_estimate_colors()
      self._setup_estimate_vecs()
      self._setup_estimate_scatterplots()
      self._setup_estimate_lcs()

  def _setup_estimate_scatterplots(self):
    """ 
    Setup the scatterplot structures that display the scatter points for 
    previous values of the different estimates
    """
    if self._do_plot_scatters:
      self._estimate_scatters = []
      for i in range(self._n_estimates):
        # Scatter plot
        if self._plot_scatters[i]:
          scat = self._ax.scatter(
            self._estimates[0, :, i], self._estimates[1, :, i], 
            self._estimates[2, :, i], facecolors=self._estimate_colors[:, :, i],
            edgecolors='none', s=10
          )
        else:
          scat = None # Place holder to keep indexing correct
        self._estimate_scatters.append(scat)

  def _setup_estimate_lcs(self):
    """
    Setup the line collection objects for storing the segments between 
    previous values of the different estimates. Certain estimates may
    not plot the segments, which is considered in constructin the relevant
    structures
    """
    if self._do_plot_lines:
      # Create 3d linecollection for storing lines between past estimates
      self._estimate_lcs = []
      for i in range(self._n_estimates):
        # Line collection
        if self._plot_lines[i]:
          lc = Line3DCollection(self._segments_from_vectors(self._estimates), lw=1)
          lc.set_color(self._estimate_colors[:-1, :, i])
          self._ax.add_collection3d(lc)
          self._estimate_lcs.append(lc)
        else:
          self._estimate_lcs.append(None) # Place holder so indexing still works

  def _setup_estimate_vecs(self):
    """
    Setup plotting structures to display vector from center of hemisphere to
    the current estimate values
    """
    self._estimate_plots = []
    colors_cycle = itertools.cycle(self._estimate_colors_strings)
    for i in range(self._n_estimates):
      self._estimate_plots.append(
          self._ax.plot([0, 0], [0, 0], [0, 0], next(colors_cycle), lw=.5)[0]
      )

  def _setup_estimate_colors(self):
    """
    Setup the structures for storing the colors of the different estimates
    as well as the change in color that occurs over values as the they become
    increasingly outdated. The resulting structure will be used in setting
    properties of the other plotting structures
    """
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
    """
    Update the structure storing all previous estimates. Also make the
    necessary updates to all plotting structures so that the display is
    updated to include new data
    """
    if len(estimates) != self._n_estimates:
      raise ValueError("Number of estimates supplied does not equal number" + \
          "of estimates supplied at instantiation")
    # Update estimate matrix
    for i, estimate in enumerate(estimates):
      self._add_frame(self._estimates[:, :, i], estimate)
    self._update_estimate_vecs()
    self._update_estimate_lcs()
    self._update_estimate_scatterplots()

  def _update_estimate_vecs(self):
    """
    Update the vector pointing from the hemisphere center to the current
    estimate value
    """
    for i, plot in enumerate(self._estimate_plots):
      plot.set_xdata([0, self._estimates[0, -1, i]])
      plot.set_ydata([0, self._estimates[1, -1, i]])
      plot.set_3d_properties([0, self._estimates[2, -1, i]])

  def _update_estimate_lcs(self):
    """
    Update the line collection objects used to store the segments between
    adjacent values for previous estimates
    """
    # Check if use has chosen to plot segments
    if self._do_plot_lines:
      for i in range(self._n_estimates):
        if self._plot_lines[i]:
          segments = self._segments_from_vectors(self._estimates[:, :, i])
          self._estimate_lcs[i].set_segments(segments)

  def _update_estimate_scatterplots(self):
    """
    Update the scatter plot objects that display the previous estimates
    """
    if self._do_plot_scatters:
      for i in range(self._n_estimates):
        if self._plot_scatters[i]:
          ptools.update_3d_scatter(
            self._estimate_scatters[i], self._estimates[:, :, i].T)

  def update(self, estimates=None, *args, **kwargs):
    """
    Update the plot given a new set of estimates. 
    :param estimates: estimates should be in the form of a list that is 
       'n_estimates' long.  Each estimate should be a length-3 numpy vector

    """
    if estimates is not None:
      self._update_estimates(estimates)
    self._update_figure()

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

  
