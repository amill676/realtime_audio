import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pa_tools.constants as consts
import mattools.mattools as mtools

class RealtimePlot(object):
  """
  Class for plotting with realtime updates
  """
  def __init__(self):
    # Don't do anything. This will allow subclasses to customize figure
    # setup without making any assumptions
    pass

  def get_figure(self):
    """
    Return the figure associated with this plot
    """
    return self._figure

  def get_axes(self):
    """
    Return the axes associated with this plot
    """
    return self._ax

  def _setup_figure(self, projection=None):
    self._figure = plt.figure()
    if projection is None:
      self._ax = self._figure.add_subplot(111)
    else:
      self._ax = self._figure.add_subplot(111, projection=projection)
    plt.show(block=False)

  def update(self, *args, **kwargs):
    self._update_figure()

  def _update_figure(self):
    plt.draw()

  def _add_frame(self, mat, frame):
    mtools.add_frame(mat, frame)

