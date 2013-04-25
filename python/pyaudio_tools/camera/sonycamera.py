__author__ = 'Adam Miller'
import urllib
from cameraconverter import CameraConverter
from cameraformatter import CameraFormatter
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
        self._formatter = CameraFormatter(url)
        self._url = url

    def face_direction(self, direction):
        """
        Turn the camera to face a specific direction
        :param direction: 3-d vector
        """
        v = mat.check_3d_vec(direction)
        pan = self._converter.get_pan(v)
        tilt = self._converter.get_tilt(v)
        command = self._formatter.absolute_pos_command(pan, tilt)
        self._send_command(command)

    def set_pan_tilt(self, pan, tilt):
        command = self._formatter.absolute_pos_command(pan, tilt)
        self._send_command(command)

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

    def _send_command(self, command):
        """
        Attempt to send the command. If it fails, a warning will be printed
        """
        try:
            urllib.urlopen(command)
        except IOError:
            sys.stderr.write('Failed to send command: %s' % command)













