import itertools
import numpy as np
import matplotlib.pyplot as plt
import pa_tools.constants as consts
from realtimeplot import RealtimePlot

class FilterPlot(RealtimePlot):
  """
  Class for plotting observation distributions and filter estimates
  over time.
  """
  def __init__(self, n_space, n_past_samples, n_estimates=0, 
                distr_cmap='bone', estimate_colors=None, *args, **kwargs):
    """
    :param n_space: number of samples in the state space discretization
    :param n_past_samples: number of past frames in time to maintain in plot
    :param n_estimates: number of estimates to plot. defaults to 0
    :param distr_cmap: color map for the likelihood distribution
    :param estimate_colors: colors for the different estimates. In the form of
                            matplotlib color formats

    All other parameters will be passed to RealtimePlot constructor
    """
    RealtimePlot.__init__(self, *args, **kwargs)
    self._n_space = n_space
    self._n_past_samples = n_past_samples
    self._n_estimates = n_estimates
    self._distr_cmap = distr_cmap
    self._estimate_colors = estimate_colors
    self._setup()

  def _setup(self):
    # Setup structures first
    self._time_space = np.arange(self._n_past_samples)
    # Holds distribution to be plotted at each time frame in each column
    self._distr_mat = np.zeros((self._n_space, self._n_past_samples))
    # Setup estimate structures
    self._setup_estimates()

    # Setup actual plot
    # Make ticks on axes invisible
    self._ax.get_xaxis().set_visible(False)
    self._ax.get_yaxis().set_visible(False)
    # Create actual plot
    self._im_2d = self._ax.imshow(self._distr_mat, vmin=0, vmax=.03, 
        cmap=self._distr_cmap, origin='lower', 
        extent=[0, self._time_space[-1], 0, np.pi], aspect='auto')
    # Setup limits
    self._ax.set_ylim(0, np.pi)
    self._ax.set_xlim(0, self._n_past_samples-1)
    # Setup lables
    self._ax.set_xlabel('time')
    self._ax.set_ylabel('DOA')
 
  def _setup_estimates(self):
    """
    Setup the matrix for holding previous estimates. Also setup the estimate
    plot structures
    """
    # Deal with estimates
    if self._n_estimates > 0:
      self._estimate_mat = np.zeros((self._n_estimates, self._n_past_samples))

      # Deal with estimate colors
      if self._estimate_colors is not None:
        if len(self._estimate_colors) != self._n_estimates:
          print "WARNING: Lenght of provided color list for estimates is " +\
              "not the same as the number of estimates indicated. Colors " +\
              "will cycle if not enough provided"
        colors = itertools.cycle(self._estimate_colors)
      else:
        colors = itertools.cycle(['b', 'r', 'k', 'g'])
      self._estimate_plots = []
      for i in range(self._n_estimates):
        self._estimate_plots.append(
          self._ax.plot(self._time_space, self._estimate_mat.T, color=next(colors), lw=1)[0]
        )

  def update(self, distr, estimates=None):
    self._update_distr(distr)
    if estimates is not None:
      self._update_estimates(estimates)
    self._update_plots()
    # Draw updated figure
    self._update_figure()

  def _update_distr(self, distr):
    distr_norm = distr - np.min(distr)
    distr_norm /= (np.sum(distr_norm) + consts.EPS)
    self._add_frame(self._distr_mat, distr_norm)

  def _update_estimates(self, estimates):
    if len(estimates) != self._n_estimates:
      raise ValueError("Number of estimates provided does not match the " + \
                        "number of estimates specified during instantiation")
    for i, estimate in enumerate(estimates):
      self._add_frame(self._estimate_mat[i, :], estimate)

  def _update_plots(self):
    self._im_2d.set_array(self._distr_mat)
    if self._n_estimates > 0:
      for i, plot in enumerate(self._estimate_plots):
        plot.set_ydata(self._estimate_mat[i, :])

