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
    def __init__(self, mic_loc, cam_loc, planes):
        """
        :param mic_loc: coordinate location of microphone array (localizer)
        :param cam_loc: coordinate location of camera
        :param planes: list of SourcePlane objects describing feasibility planes
        """
        self._process_locations(mic_loc, cam_loc)
        self._process_planes(planes)

    def _process_locations(self, mic_loc, cam_loc):
        """
        Verify location inputs and setup associated member variables
        """
        if len(mic_loc.shape) != 1 or len(cam_loc.shape) != 1:
            raise ValueError('mic_loc and cam_loc should be numpy vectors. (Should have one axis)')
        if len(mic_loc) != 3 or len(cam_loc) != 3:
            raise ValueError('mic_loc and cam_loc should be 3 dimensional vectors')
        self._mic_loc = mic_loc
        self._cam_loc = cam_loc

    def _process_planes(self, planes):
        """
        Verify planes given, and setup associated member variable
        """
        if type(planes) is not list:
            raise ValueError('planes should be a list of SourcePlane objects')
        if len(planes) < 1:
            raise ValueError('planes must contain at least one SourcePlane')
        for plane in planes:
            if type(plane) is not SourcePlane:
                raise ValueError('objects in planes should be of type SourcePlane')
        self._planes = planes

    def get_source_loc(self, dir_from_mic):
        """
        Get the location of the source using the estimated direction of
        the source from the perspective of teh microphone array (localizer)
        :param dir_from_mic: 3-d vector as numpy array. Direction from localizer
        :returns: Absolute coordinates of the source as 3-d vector (numpy array)
        """
        # Setup line along projected source direction
        offset = self._mic_loc
        grad = dir_from_mic
        # Track best estimate so far
        estimate = np.array([0, 0, 0])
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
        vec = location - self._cam_loc
        # Normalize
        norm = np.sum(vec ** 2) ** .5
        if norm != 0:
            vec /= norm
        return vec





