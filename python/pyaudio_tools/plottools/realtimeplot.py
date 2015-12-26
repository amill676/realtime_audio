import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pa_tools.constants as consts
import mattools.mattools as mtools

class RealtimePlot(object):
  """
  Class for plotting with realtime updates
  """
  def __init__(self, title='', projection=None):
    self._title = title
    self._setup_figure(projection)

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
    """
    Setup this object's underlying figure. If a projection is given, use it.
    """
    self._figure = plt.figure()
    if projection is None:
      self._ax = self._figure.add_subplot(111)
    else:
      self._ax = self._figure.add_subplot(111, projection=projection)
    self._ax.set_title(self._title)
    plt.show(block=False)

  def update(self, *args, **kwargs):
    self._update_figure()

  def _update_figure(self):
    self._figure.canvas.draw()

  def _add_frame(self, mat, frame):
    mtools.add_frame(mat, frame)

