from mattools import mattools as mat

__author__ = 'Adam Miller'
from pa_tools.audiolocalizer import AudioLocalizer
import numpy as np
import math
import constants as consts
import sys
from distributionlocalizer import DistributionLocalizer

class TrackingLocalizer(DistributionLocalizer):

  def __init__(self, mic_positions, search_space, dft_len=512, sample_rate=44100,
                     n_theta=20, n_phi=1):
    """


    """
    DistributionLocalizer.__init__(self, mic_positions, dft_len, sample_rate,
                                   n_theta, n_phi)
    self._process_search_space(search_space)

    def _process_search_space(self, search_space):
      self._search_space = search_space
      self._planes =- self._search_space.get_planes()
      self._tracking_plane = self._planes[0]

