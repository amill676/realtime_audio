__author__ = 'Adam Miller'
import unittest
from camera import CameraFormatter
from camera.constants import *


class CameraFormatterTest(unittest.TestCase):

    def setUp(self):
        self.form = CameraFormatter('http:/hi/')
        pass

    def testPanHex0(self):
        pan = 0
        hex = self.form.get_pan_hex(pan)
        self.assertEquals(0, 0)

    def testPanHexMax(self):
        pan = 170
        pan_hex = self.form.get_pan_hex(pan)
        self.assertEquals(pan_hex, MAX_PAN_HEX)

    def testPanHexMin(self):
        pan = -170
        pan_hex = self.form.get_pan_hex(pan)
        self.assertEquals(pan_hex, MIN_PAN_HEX)

    def testTilt90(self):
        tilt = -90
        tilt_val = self.form.get_tilt_hex(tilt)
        self.assertEquals(tilt_val, MIN_TILT_HEX)

    def testTilt25(self):
        tilt = 25
        tilt_val = self.form.get_tilt_hex(tilt)
        self.assertEquals(tilt_val, MAX_TILT_HEX)

    def testOverTilt(self):
        tilt = 26
        tilt_val = self.form.get_tilt_hex(tilt)
        self.assertEquals(tilt_val, MAX_TILT_HEX)

    def tearDown(self):
        pass
