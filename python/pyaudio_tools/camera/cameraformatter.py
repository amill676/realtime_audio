__author__ = 'Adam Miller'
from constants import *


class CameraFormatter(object):
    """
    Class for creating and formatting proper commands for
     Sony Camera communication
    """
    def __init__(self, url):
        """
        :param url: url of camera
        """

    def absolute_pos_command(self, pan, tilt):
        """
        Get the command to send to camera to go to a specified pan
        and tilt angle
        """
        pan = self._check_pan(pan)
        tilt = self._check_tilt(tilt)

    def _check_pan(self, pan):
        """
        Ensure pan is within range
        :returns: pan after adjusting to be within bounds
        """
        if pan > MAX_PAN_DEGREE:
            return MAX_PAN_DEGREE
        if pan < MIN_PAN_DEGREE:
            return MIN_PAN_DEGREE
        return float(pan)

    def _check_tilt(self, tilt):
        """
        Ensure tilt is within range
        :returns: tilt after being adjusted to be within bounds.
        """
