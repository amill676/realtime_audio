from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pa_tools.constants as consts

TEXTWIDTH = 469.0
SUBFIG_SCALE = .45
THIRDFIG_SCALE = .32

def setup_figsize(pts, factor=1.0):
    """
    Takes number of pts across for figure to occupy and
    sets up the figure size to correspond to that size

    see:
    http://damon-is-a-geek.com/publication-ready-the-first-time-beautiful-reproducible-plots-with-matplotlib.html
    """
    #WIDTH = 350.0  # the number latex spits out
    WIDTH = pts
    FACTOR = factor
    fig_width_pt  = WIDTH * FACTOR

    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims      = [fig_width_in, fig_height_in] # fig dims as a list
    print fig_dims
    rcParams['figure.figsize'] = fig_dims

def setup_fullpage_figsize():
  setup_figsize(TEXTWIDTH, 1.0)

def setup_halfpage_figsize():
  setup_figsize(TEXTWIDTH, SUBFIG_SCALE)
  plt.locator_params(nbins=2)

def setup_thirdpage_figsize():
  setup_figsize(TEXTWIDTH, THIRDFIG_SCALE)
  plt.locator_params(nbins=2)

def get_fullpage_axis(fig):
  ax = fig.add_subplot(111)
  ax.locator_params(nbins=6)
  return ax

def get_halfpage_axis(fig):
  ax = fig.add_subplot(111)
  ax.locator_params(nbins=4)
  return ax

def update_3d_scatter(scat, newlocs):
  """
  Update scatter plot locations given a set of new locations
  :param scat: scatter plot handle to use
  :param newlocs: n x 3 array with n new locations
  """
  scat._offsets3d = (newlocs[:, 0], newlocs[:, 1], newlocs[:, 2])

