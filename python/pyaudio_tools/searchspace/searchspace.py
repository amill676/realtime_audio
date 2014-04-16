__author__ = 'Adam Miller'
from sourceplane import SourcePlane
import mattools as tools
import numpy as np


class SearchSpace(object):
    """
    Class for representing the search space containing a source
    localizer, a camera, and 'feasibility planes'.

    The feasibility planes are 3-d planes that are each viewed
    as a collection of feasible locations for the source. In turn,
    if the direction of the source is known from the perspective
    of the source localizer, we can define some objective function
    over all possible source locations (the union of feasibility
    planes) and then choose the source location that will minimize
    that objective function. For now, the objective function used
    will be the distance to the source, with a few minor restrictions

    Once the exact source location has been estimated, we can define
    its location with respect to the camera and then all information
    is determined.
    """
    def __init__(self, mic_loc, cam_loc, planes, mic_forward, mic_above):
        """
        :param mic_loc: coordinate location of microphone array (localizer)
        :param cam_loc: coordinate location of camera
        :param planes: list of SourcePlane objects describing feasibility planes
        :param mic_forward: the vector corresponding to the postive y direction
                            for the mic, in world frame
        :param mic_above: the vector corresponding to the postive z direction
                          for the mic, in world frame
        """
        self._process_locations(mic_loc, cam_loc)
        self._process_planes(planes)
        self._setup_transform(mic_forward, mic_above)

    def _process_locations(self, mic_loc, cam_loc):
        """
        Verify location inputs and setup associated member variables
        """
        self._mic_loc = tools.check_3d_vec(mic_loc)
        self._cam_loc = tools.check_3d_vec(cam_loc)

    def _setup_transform(self, mic_forward, mic_above):
        mic_forward = tools.check_3d_vec_normalize(mic_forward)
        mic_above = tools.check_3d_vec_normalize(mic_above)
        mic_right = np.cross(mic_forward, mic_above)
        # Setup basis matrix for mic coordinate system.
        self._transform_mat = np.array([mic_right, mic_forward, mic_above])

    def _process_planes(self, planes):
        """
        Verify planes given, and setup associated member variable
        """
        if type(planes) is not list:
            raise ValueError('planes should be a list of SourcePlane objects')
        if len(planes) < 1:
            raise ValueError('planes must contain at least one SourcePlane')
        #for plane in planes: We'll duck type I suppose...
        #    if type(plane) is not SourcePlane:
        #        raise ValueError('objects in planes should be of type SourcePlane')
        self._planes = planes

    def get_mic_loc(self):
        return self._mic_loc

    def get_cam_loc(self):
        return self._cam_loc

    def get_planes(self):
        return self._planes

    def get_source_loc(self, dir_from_mic):
        """
        Get the location of the source using the estimated direction of
        the source from the perspective of teh microphone array (localizer)
        :param dir_from_mic: 3-d vector as numpy array. Direction from localizer
                             in frame of localizer
        :returns: Absolute coordinates of the source as 3-d vector (numpy array)
        """
        # Setup line along projected source direction
        offset = self._mic_loc
        grad = self._transform_mat.dot(dir_from_mic)
        # Track best estimate so far
        estimate = None
        estimate_dist = np.inf
        for plane in self._planes:
            location = plane.line_intersection(grad, offset)
            # If plane does not intersect with DOA vector
            if location is None:
                continue
            dist = np.linalg.norm(location - offset, 2)
            # Ensure that location is in direction of DOA, not opposite
            if grad.dot(location - offset) > 0 and dist < estimate_dist:
                estimate = location
                estimate_dist = dist
        return estimate

    def get_camera_dir(self, dir_from_mic):
        """
        Get the direction to the source from the camera's point of view
        :param dir_from_mic: 3-d vector numpy array that is the DOA estimate
                             from the mic's perspective
        :returns: direction to source from camera as 3-d vector (numpy array)
        """
        location = self.get_source_loc(dir_from_mic)
        print "Direction: %s, location: %s" % (dir_from_mic, location)
        if location is None:
          return None
        vec = location - self._cam_loc
        # Normalize
        norm = np.sum(vec ** 2) ** .5
        if norm != 0:
            vec /= norm
        return vec





