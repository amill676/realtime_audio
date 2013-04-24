__author__ = 'Adam Miller'
import urllib
from cameraconverter import CameraConverter
import mattools as mat
import sys
import numpy as np
import math
EPS = sys.float_info.epsilon


class SonyCamera(object):
    """
    Class for interacting with Sony Camera
    """

    def __init__(self, url, forward, above):
        """
        :param url: URL for connecting to camera
        :param forward: vector corresponding to the direction pointed by
                        the camera at 0 pan and 0 tilt. This should be in
                        the coordinate system of the camera's containing
                        search space. Should be orthogonal to 'above'.
        :param above: vector that represents the camera's panning axis.
                      i.e. the vector pointing up through the top of the camera
                      and normal to the panning plane. This should be in the coordinate
                      system of the camera's containing search space (same
                      coordinate system as the forward vector). Should be
                      orthogonal to 'above'.
        """
        self._converter = CameraConverter(forward, above)
        self._url = url

    def connect(self):
        """
        Attempt to connect to specified URL
        :returns: True if connection succeeds, False otherwise
        """
        try:
            self._conn = urllib.urlopen(self._url)
        except IOError:
            return False
        return True











