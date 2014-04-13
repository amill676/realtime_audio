__author__ = 'Adam Miller'
import numpy as np
import mattools as tools


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

    def get_normal(self):
        return self._normal

    def get_offset(self):
        return self._offset

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
        lin_offset = tools.check_3d_vec(offset)
        grad = tools.check_3d_vec(grad)
        # Check if line and plane are parallel
        if abs(np.max(grad.dot(self._normal))) < 1e-9:
            return None
        t = (self._normal.dot(self._offset - lin_offset)) / (self._normal.dot(grad))
        if t < 0:
            return None
        return lin_offset + t * grad

    def _verify_params(self, normal, offset):
        """
        Ensure vector parameters passed to init are valid
        """
        self._normal = tools.check_3d_vec(normal)
        self._normal /= tools.norm2(self._normal)
        self._offset = tools.check_3d_vec(offset)



