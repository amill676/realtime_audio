__author__ = 'Adam Miller'
import unittest
import numpy as np
from camera import CameraConverter

class CameraConverterTest(unittest.TestCase):

    def setUp(self):
        # Standard coordinates
        self._forward = np.array([1, 0, 0])
        self._above = np.array([0, 0, 1])
        self._conv = CameraConverter(self._forward, self._above)
        # Alternate coordinates
        self._forward2 = np.array([1, 1, 1])
        self._above2 = np.array([-1, -1, 2])
        self._conv2 = CameraConverter(self._forward2, self._above2)

    def testNonOrthogonalVectors(self):
        forward = np.array([1, 1, 1])
        above = np.array([1, 1, 0])
        self.assertRaises(ValueError, CameraConverter, forward, above)

    def testZeroLengthVectors(self):
        forward = np.array([0, 0, 0])
        above = np.array([1, 0, 0])
        self.assertRaises(ValueError, CameraConverter, forward, above)

    def testPanX(self):
        direction = self._forward
        pan = self._conv.get_pan(direction)
        self.assertEquals(0, pan)

    def testPanY(self):
        direction = np.array([0, 1, 1])
        pan = self._conv.get_pan(direction)
        self.assertLessEqual(90, pan)

    def testPanNegY(self):
        direction = np.array([0, -1, 1])
        pan = self._conv.get_pan(direction)
        self.assertEquals(-90, pan)

    def testPan111(self):
        direction = np.array([1, 1, 1])
        pan = self._conv.get_pan(direction)
        self.assertLessEqual(abs(45 - pan), 1e-9)

    def testOverPan(self):
        direc = np.array([-1, .05, 2])
        pan = self._conv.get_pan(direc)
        self.assertEquals(170, pan)

    def testNegOverPan(self):
        direc = np.array([-1, -.05, 2])
        pan = self._conv.get_pan(direc)
        self.assertEquals(-170, pan)

    def testTiltX(self):
        direc = np.array([1, 0, 0])
        tilt = self._conv.get_tilt(direc)
        self.assertEquals(0, tilt)

    def testTiltY(self):
        direc = np.array([0, 1, 0])
        tilt = self._conv.get_tilt(direc)
        self.assertEquals(0, tilt)

    def testTilt45(self):
        direc = np.array([1, 0, -1])
        tilt = self._conv.get_tilt(direc)
        self.assertLessEqual(abs(tilt - 45), 1e-9)

    def testTilt90(self):
        direc = np.array([0, 0, -1])
        tilt = self._conv.get_tilt(direc)
        self.assertEquals(90, tilt)

    def testOverTilt(self):
        direc = np.array([0, 0, 1])
        tilt = self._conv.get_tilt(direc)
        self.assertEquals(-25, tilt)

    def testPanX2(self):
        direction = self._forward2
        pan = self._conv2.get_pan(direction)
        self.assertEquals(0, pan)

    def testPanY2(self):
        direction = np.array([-1, 1, 0])
        pan = self._conv2.get_pan(direction)
        self.assertEquals(90, pan)

    def testPanNegY2(self):
        direction = np.array([1, -1, 0])
        pan = self._conv2.get_pan(direction)
        self.assertEquals(-90, pan)

    def testOverPan2(self):
        direc = np.array([-1, -.95, -1])
        pan = self._conv2.get_pan(direc)
        self.assertEquals(170, pan)

    def testNegOverPan2(self):
        direc = np.array([-1, -1.05, -1])
        pan = self._conv2.get_pan(direc)
        self.assertEquals(-170, pan)

    def testTiltX2(self):
        direc = self._forward2
        tilt = self._conv2.get_tilt(direc)
        self.assertEquals(0, tilt)

    def testTiltY2(self):
        direc = np.array([-1, 1, 0])
        tilt = self._conv2.get_tilt(direc)
        self.assertEquals(0, tilt)

    def testTiltAngle2(self):
        direc = np.array([1, 1, 0])
        tilt = self._conv2.get_tilt(direc)
        self.assertLessEqual(abs(tilt - 35.2644), 1e-3)

    def testTilt902(self):
        direc = np.array([1, 1, -2])
        tilt = self._conv2.get_tilt(direc)
        self.assertEquals(90, tilt)

    def testOverTilt2(self):
        direc = np.array([0, 0, 1])
        tilt = self._conv2.get_tilt(direc)
        self.assertEquals(-25, tilt)

    def tearDown(self):
        pass
