__author__ = 'Adam Miller'
import unittest
import numpy as np
from searchspace import SourcePlane


class SourcePlaReTest(unittest.TestCase):

    def setUp(self):
        pass

    def testValidSetup(self):
        normal = np.array([1, 2, 3])
        offset = np.array([3, 1, 1])
        plane = SourcePlane(normal, offset)

    def testInvalidNormal(self):
        normal = np.array([1, 2])
        offset = np.array([3, 1, 1])
        self.assertRaises(ValueError, SourcePlane, normal, offset)

    def testInvalidOffset(self):
        normal = np.array([1, 2, 3])
        offset = np.array([3, 1, 1, 4])
        self.assertRaises(ValueError, SourcePlane, normal, offset)

    def testMatrixNormal(self):
        normal = np.array([[1, 2, 3]])
        offset = np.array([3, 1, 1])
        self.assertRaises(ValueError, SourcePlane, normal, offset)

    def testMatrixOffset(self):
        normal = np.array([1, 2, 3])
        offset = np.array([[3, 1, 1]])
        self.assertRaises(ValueError, SourcePlane, normal, offset)

    def testLineIntersection(self):
        normal = np.array([0, 0, 1])
        offset = np.array([0, 0, 2.5])
        plane = SourcePlane(normal, offset)
        lin_off = np.array([0, 0, 0])
        grad = np.array([1, 1, 1])
        intersection = plane.line_intersection(lin_off, grad)
        self.assertListEqual(list(intersection), [2.5, 2.5, 2.5])

    def testLineIntersection2(self):
        normal = np.array([2, 2, 2])
        offset = np.array([0, 0, 7])
        plane = SourcePlane(normal, offset)
        lin_off = np.array([0, 0, -2])
        grad = np.array([-1, -1, -2])
        intersection = plane.line_intersection(lin_off, grad)
        self.assertListEqual(list(intersection), [2.25, 2.25, 2.5])

    def tearDown(self):
        pass
