__author__ = 'Adam Miller'
import numpy as np
import mattools as tools
import pa_tools.constants as consts

from sourceplane import SourcePlane

class OrientedSourcePlane(SourcePlane):
  """
  Class to represent a sourceplane that has a sense of orientation upon the
  plane. This is useful for cases where a sources location must be described
  by a coordinate system within the plane.
  """
  def __init__(self, normal, up, offset):
        """
        :param normal: numpy 3-d vector representing normal vector to plane
        :param up: numpy 3-d vector representing the up direction upon the
                   plane. Must be within the plane, and thus orthogonal to the
                   given normal argument
        :param offset: numpy 3-d vector representing offset from origin to plane
        """
        self._verify_params(normal, up, offset)
        self._setup_transforms()

  def get_up(self):
    return self._up

  def to_plane_coordinates(self, x):
    """
    Convert a given vector x into a vector y of coordinates within the plane.
    The new vector will be such that:
      x = y_1*right + y_2*up + y_3*normal + offset
    where up and normal are the unit coordinate vectors provided, and right 
    is unit coordinate vector given by up (cross) normal.
    Note that the third dimension (y_3) will always be 0, since the point must
    lie on the plane

    :param x: 3-d numpy vector in word coordinates
    :returns: equivalent 3-d numpy vector in plane coordinates
    """
    shifted = tools.check_3d_vec(x) - self._offset
    return self._transform_mat.T.dot(shifted)

  def from_plane_coordinates(self, y):
    """
    Peforms inverse operation as to_plane_coordinates(x). 
    :param y: 3-d numpy vector in plane coordinates. Third component must be 0
    :returns: equivalent 3-d numpy vector in world coordinates
    """
    y = tools.check_3d_vec(y)
    if np.abs(y[2]) > consts.EPS:
      raise ValueError("Input vector's third component must be 0")
      
    return self._transform_mat.dot(y) + self._offset

  def get_transform_mat(self):
    """ 
    Get the transformation matrix W such that for a vector x in world 
    coordinates and a vector y in plane coordinates, x = Wy + offset
    """
    return self._transform_mat
    
  def _verify_params(self, normal, up, offset):
    """
    Ensure vector parameters passed to init are valid
    """
    self._normal = tools.check_3d_vec(normal)
    self._normal /= tools.norm2(self._normal)
    self._up = tools.check_3d_vec(up)
    self._up /= tools.norm2(self._up)
    self._offset = tools.check_3d_vec(offset)
    # Get third coordinate vector
    self._right = np.cross(self._up, self._normal)
    self._right /= tools.norm2(self._right)

  def _setup_transforms(self):
    """
    Setup necessary structures for performing transforms easily. Namely,
    setup a transform matrix such that for a vector x in world coordinates
    and a vector y in plane coordinates, x = Wy + offset
    """
    # Columns will be right, up, normal in that order
    self._transform_mat = np.array([self._right, self._up, self._normal]).T






