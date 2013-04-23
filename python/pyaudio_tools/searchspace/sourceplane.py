__author__ = 'Adam Miller'
import numpy as np


class SourcePlane(object):
    """
    Class to represent the plane on which a source may lie within
    a search space
    """
    def __init__(self, normal, offset):
        """
        :param normal: numpy 3-d vector representing normal vector to plane
        :param offset: numpy 3-d vector representing offset from origin to plane
        """
        self._verify_params(normal, offset)

    def line_intersection(self, grad, offset):
        """
        Find point at which the line described by the given parameters
        will intersect the plane. This will be a 3 dimensional coordinate
        vector

        Line will be of the form l(t) = offset + grad * t, where
        offset and grad are both 3-dimensional vectors

        For plane defined as 0 = n.dot(x - m), and line as l(t) = a + b*t
        where x, m, a, b are 3-d vectors, we can calculate the intersection as
        intersection = a + b * (n.dot(m - a) / n.dot(b)) which follows from
        solving 0 = n.dot(l(t) - m) for t

        :param offset: 3-dimensional numpy vector describing line offset
        :param grad: 3-dimensional numpy vector describing line gradient
        :returns: 3-dimensional numpy vector describing intersection coordinate.
                  If the line and plane do not intersect, None is returned
        """
        lin_offset = self._check_vec(offset)
        grad = self._check_vec(grad)
        print grad
        print self._normal
        # Check if line and plane are parallel
        if abs(np.max(grad.dot(self._normal))) < 1e-9:
            return None
        t = (self._normal.dot(self._offset - lin_offset)) / (self._normal.dot(grad))
        return lin_offset + t * grad

    def _verify_params(self, normal, offset):
        """
        Ensure vector parameters passed to init are valid
        """
        self._normal = self._check_vec(normal)
        self._offset = self._check_vec(offset)

    def _check_vec(self, vec):
        """
        Ensure that the input is a 3-d vector and has float type.
        :returns: The float version of the vector if it has proper size. If already
                    float, no copy is made
        """
        if len(vec.shape) != 1:
            raise ValueError("vectors must be a numpy vector array (should have only one axis)")
        if len(vec) != 3:
            raise ValueError("vectors must be 3 dimensional")
        return self._to_float(vec)

    def _to_float(self, arr):
        """
        Return the array with a floating point dtype. If the array
        already has such a dtype, it will not be copied, but simply
        returned
        :param arr: numpy array to be converted
        """
        dtype = arr.dtype
        if not (dtype is np.float16 or dtype is np.float32 or dtype is np.float64 or
                    dtype is np.complex64 or dtype is np.complex128):
            return arr.astype(np.float64)  # Same as np.float
        return arr

