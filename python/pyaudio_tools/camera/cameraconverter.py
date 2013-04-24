__author__ = 'Adam Miller'
import numpy as np
import mattools as mat
import sys
import math
EPS = sys.float_info.epsilon
from constants import *


class CameraConverter(object):
    """
    Class for taking care of coordinate transforms and conversion
    of quantities necessary for controlling movement of camera
    """
    def __init__(self, forward, above):
        self._setup_transform(forward, above)

    def get_tilt(self, direction):
        """
        :param direction: Direction to find tilt angle for
        :returns: tilt angle in degrees
        """
        # Noramlize
        direction = mat.check_3d_vec(direction)
        if mat.norm2(direction) < EPS:
            return 0.  # No tilt in zero vec...
        direction /= mat.norm2(direction)

        # Transform into native coordinates
        direction = self._transform(direction)

        # Get tilt angle
        above = np.array([0, 0, 1.])  # Above in positive z direction
        dot_prod = -1. * direction.dot(above)
        if abs(dot_prod) > 1:  # Small floating point errors can give domain erros in acos
            dot_prod = mat.sign(dot_prod)
        tilt = math.acos(dot_prod)
        tilt = 90 - mat.to_degrees(tilt)  # Now 0 corresponds to x-y plane, 90 to -z
        return max(tilt, MIN_TILT_DEGREE + TILT_CENTER_DEGREE)  # Cannot go above 25 degrees

    def get_pan(self, direction):
        """
        Get pan angle in degrees for a given direction. The given direction
        should be in the same coordinate system as the above and forward
        directions that were given in initialization. That is, they should
        not be transformed
        :param direction: 3-d vector
        :returns: pan angle in degrees
        """
        # Check input
        direction = mat.check_3d_vec(direction)

        # Transform into native coordinates
        direction = self._transform(direction)
        direction *= np.array([1, 1, 0])  # Project onto xy plane
        if mat.norm2(direction) < EPS:  # No component in xy plane
            return 0
        direction /= mat.norm2(direction)

        # Get pan angle
        forward = np.array([1, 0, 0])  # Forward is positive x direction
        dot_prod = direction.dot(forward)
        if abs(dot_prod) > 1:  # Small floating point errors can give domain erros in acos
            dot_prod = mat.sign(dot_prod)
        pan = math.acos(dot_prod)  # x.dot(dir) = cos(pan)
        pan = min(mat.to_degrees(pan), MAX_PAN_DEGREE / 2)
        # Determine whether should be in [0, pi] or [0, -pi]
        y = np.array([0, 1, 0])
        pan *= mat.sign(y.dot(direction))

        return pan

    def _setup_transform(self, forward, above):
        """
        Setup the transformation matrix that will be used to transform
        given coordinates into the native coordinate system.

        Note that forward and above should be orthogonal vectors.

        In the native coordinate system, the forward vector will correspond
        to the positive x direction and the above vector to the
        positive z direction. So denote x and z as the unit vectors in
        those respective directions. Then we take y to be z cross x.
        Now we want to find transformation matrix R such that given direction
        d in the search space coordinate system, we can apply R to d and
        get w - the same direction in the native coordinate system.
        We assume the search coordinate system are standard i, j, k vectors.
        This is ok since the forward and above vectors are defined in those coordinates
        Then the linear transformation matrix for the search coordinate
        system is the identity matrix. So we now seek to solve
        I*d = A*w, where A's columns consist of x, y, z. So to get w,
        we have inv(A)*d = w. Since x, y, z are orthogonal, this is the
        same as A.T*d = w, and thus R = A.T
        """
        # Check inputs
        forward = mat.check_3d_vec(forward)
        above = mat.check_3d_vec(above)
        if mat.norm2(forward) < EPS or mat.norm2(above) < EPS:
            raise ValueError("forward and above vectors must not be length 0")
        if abs(forward.dot(above)) > EPS:
            raise ValueError("forward and above vectors must be orthogonal")

        # Normalize vectors
        x = forward / mat.norm2(forward)
        z = above / mat.norm2(above)
        y = np.cross(z, x)
        y /= mat.norm2(y)  # just in case

        # Make matrix
        self._trans_mat = np.array([x, y, z])

    def _transform(self, vec):
        """
        Transform the given vector into the native coordinates
        """
        return self._trans_mat.dot(vec)
