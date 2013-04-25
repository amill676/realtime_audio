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
        self._url = url

    def absolute_pos_command(self, pan, tilt, pan_speed=MAX_PAN_SPEED, tilt_speed=MAX_TILT_SPEED):
        """
        Get the command to send to camera to go to a specified pan
        and tilt angle
        """
        # Clean inputs
        pan_speed = self._check_bounds(float(pan_speed), MIN_PAN_SPEED, MAX_PAN_SPEED)
        tilt_speed = self._check_bounds(float(tilt_speed), MIN_TILT_SPEED, MAX_TILT_SPEED)


        # Get hex values for angles
        pan_hex = self.get_pan_hex(pan)
        tilt_hex = self.get_tilt_hex(tilt)

        # Create formatted string
        return "%s%s%s%02x%02x%s%sFF" % (self._url, VISCA_COMMAND_STR, ABSOLUTE_POS_STR,
                                       pan_speed, tilt_speed, self._hex_to_string(pan_hex),
                                       self._hex_to_string(tilt_hex))

    def get_tilt_hex(self, tilt):
        """
        Get the hex value for tilt value in degrees
        :param tilt: tilt value in degrees
        :returns: tilt value in hex
        """
        tilt = self._check_tilt(tilt)
        # Note that as degrees become negative, hex will increase
        val = MIN_TILT_HEX + (tilt - MIN_TILT_DEGREE) * \
               (MAX_TILT_HEX - MIN_TILT_HEX) / (MAX_TILT_DEGREE - MIN_TILT_DEGREE)
        return int(val)

    def get_pan_hex(self, pan):
        """
        Get the hex value for a pan value in degrees
        :param pan: pan value in degrees
        :returns: pan value in hex
        """
        pan = self._check_pan(float(pan))
        val = float(MIN_PAN_HEX) + float(pan - MIN_PAN_DEGREE) * \
                float(MAX_PAN_HEX - MIN_PAN_HEX) / float(MAX_PAN_DEGREE - MIN_PAN_DEGREE)
        return -1 * int(val)  # Degree sign is flipped in camera's default coordinates

    def _check_pan(self, pan):
        """
        Ensure pan is within range
        :param pan: pan value in degrees
        :returns: pan after adjusting to be within bounds
        """
        return self._check_bounds(float(pan), MIN_PAN_DEGREE, MAX_PAN_DEGREE)

    def _check_tilt(self, tilt):
        """
        Ensure tilt is within range
        :param tilt: tilt value in degrees
        :returns: tilt after being adjusted to be within bounds.
        """
        return self._check_bounds(float(tilt), MIN_TILT_DEGREE, MAX_TILT_DEGREE)

    def _check_bounds(self, val, min_val, max_val):
        """
        Ensure a given value is within set bounds
        :param val: value to check
        :param max_val: max acceptable value
        :param min_val: min acceptable value
        :returns: version of the value within bounds
        """
        if val > max_val:
            return max_val
        if val < min_val:
            return min_val
        return val

    def _hex_to_string(self, hex_val):
        """
        Takes a 4 digit hex value and returns the string version
        interspersed with 0's
        e.g. ABCD -> "0A0B0C0D"
        :param hex_val: hex value
        :returns: string as described above
        """
        hex_str = "%04x" % (hex_val & 0xFFFF)
        with_zeros = "0%c0%c0%c0%c" % (hex_str[0], hex_str[1], hex_str[2], hex_str[3])
        return with_zeros



