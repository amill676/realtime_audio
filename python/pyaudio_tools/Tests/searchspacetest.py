__author__ = 'Adam Miller'
import unittest
import numpy as np
from searchspace import SourcePlane
from searchspace import SearchSpace


class SearchSpaceTest(unittest.TestCase):

    def setUp(self):
        pass

    def testInit(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        planes = [SourcePlane(offset, normal)]
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1, 1])
        space = SearchSpace(mic_loc, cam_loc, planes)

    def testInvalidMicLoc(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        planes = [SourcePlane(offset, normal)]
        mic_loc = np.array([0, 0])
        cam_loc = np.array([1, 1, 1])
        self.assertRaises(ValueError, SearchSpace, mic_loc, cam_loc, planes)

    def testInvalidCamLoc(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        planes = [SourcePlane(offset, normal)]
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1])
        self.assertRaises(ValueError, SearchSpace, mic_loc, cam_loc, planes)

    def testInvalidMicLocArray(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        planes = [SourcePlane(offset, normal)]
        mic_loc = np.array([[0, 0, 0]])
        cam_loc = np.array([1, 1, 1])
        self.assertRaises(ValueError, SearchSpace, mic_loc, cam_loc, planes)

    def testInvalidCamLocArray(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        planes = [SourcePlane(offset, normal)]
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([[1, 1, 1]])
        self.assertRaises(ValueError, SearchSpace, mic_loc, cam_loc, planes)

    def testInvalidEmptyPlanes(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        planes = []
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1, 1])
        self.assertRaises(ValueError, SearchSpace, mic_loc, cam_loc, planes)

    def testInvalidPlanes(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        planes = SourcePlane(offset, normal)
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1, 1])
        self.assertRaises(ValueError, SearchSpace, mic_loc, cam_loc, planes)

    def testGetSourceLoc1Plane(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        planes = [SourcePlane(normal, offset)]
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1, 1])
        space = SearchSpace(mic_loc, cam_loc, planes)
        direction = np.array([-1, 1, 1])
        loc = space.get_source_loc(direction)
        self.assertListEqual(list(loc), [-5, 5, 5])

    def testGetSourceLoc2Plane(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        normal2 = np.array([0, 0, 1])
        offset2 = np.array([0, 0, 3])
        planes = [SourcePlane(normal, offset), SourcePlane(normal2, offset2)]
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1, 1])
        space = SearchSpace(mic_loc, cam_loc, planes)
        direction = np.array([-1, 1, 1])
        loc = space.get_source_loc(direction)
        self.assertListEqual(list(loc), [-3, 3, 3])

    def testGetSourceLoc2PlaneBehind(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        normal2 = np.array([0, 0, 1])
        offset2 = np.array([0, 0, -1])
        planes = [SourcePlane(normal, offset), SourcePlane(normal2, offset2)]
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1, 1])
        space = SearchSpace(mic_loc, cam_loc, planes)
        direction = np.array([-1, 1, 1])
        loc = space.get_source_loc(direction)
        self.assertListEqual(list(loc), [-5, 5, 5])

    def testGetSourceLoc2Plane2(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        normal2 = np.array([0, 0, 1])
        offset2 = np.array([0, 0, 6])
        planes = [SourcePlane(normal, offset), SourcePlane(normal2, offset2)]
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1, 1])
        space = SearchSpace(mic_loc, cam_loc, planes)
        direction = np.array([-1, 1, 1])
        loc = space.get_source_loc(direction)
        self.assertListEqual(list(loc), [-5, 5, 5])

    def testGetSourceLoc3Plane(self):
        normal = np.array([0, 1, 0])
        offset = np.array([0, 5, 0])
        normal2 = np.array([0, 0, 1])
        offset2 = np.array([0, 0, 6])
        normal3 = np.array([1, 0, 0])
        offset3 = np.array([-1, 0, 0])
        planes = [SourcePlane(normal, offset), SourcePlane(normal2, offset2), SourcePlane(normal3, offset3)]
        mic_loc = np.array([0, 0, 0])
        cam_loc = np.array([1, 1, 1])
        space = SearchSpace(mic_loc, cam_loc, planes)
        direction = np.array([-1, 1, 1])
        loc = space.get_source_loc(direction)
        self.assertListEqual(list(loc), [-1, 1, 1])

    def tearDown(self):
        pass
