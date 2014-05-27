from mattools import mattools as mat

__author__ = 'Adam Miller'
from pa_tools.audiolocalizer import AudioLocalizer
import numpy as np
import math
import constants as consts
import sys
from distributionlocalizer import DistributionLocalizer

class TrackingLocalizer(DistributionLocalizer):

  def __init__(self, search_space, *args, **kwargs):
    """
    :param search_space: SearchSpace object that describes the search space in
                         which the microphone is located.
    """
    DistributionLocalizer.__init__(self, *args, **kwargs)
    self._process_search_space(search_space)

  def _process_search_space(self, search_space):
    self._search_space = search_space
    self._planes = self._search_space.get_planes()
    self._tracking_plane = self._planes[0]

