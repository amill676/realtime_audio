from pa_tools.stftmanager import StftManager


__author__ = 'adamjmiller'
import unittest
import numpy as np



class StftManagerTest(unittest.TestCase):
    """
    Tester for StftManager class
    """

    def setUp(self):
        self.data_type = np.float32
        self.window_len = 16
        self.dft_len = 16
        self.hop_len = self.window_len  # No overlap
        self.use_window = False  # No hann
        self.n_channels = 1
        self.stft_dft = StftManager(dft_length=self.dft_len,
                                    window_length=self.window_len,
                                    hop_length=self.hop_len,
                                    use_window_fcn=self.use_window,
                                    n_channels=self.n_channels,
                                    dtype=self.data_type)
        print "Beginning"

    def testConstructor(self):
        StftManager()

    def testData(self):
        data = np.array([1, 0, 1, 0], dtype=np.float32)
        self.stft_dft.performStft(data)

    def testOutData(self):
        data = np.array(np.ones(self.window_len), dtype=np.float32)
        self.stft_dft.performStft(data)
        data1 = np.array(np.zeros(self.window_len), dtype=np.float32)
        self.stft_dft.performIStft(data1)
        #self.stft.performIStft(data1)
        for i in range(len(data)):
            print i
            print "data: " + str(data[i])
            print "data1:" + str(data1[i])
            self.assertEquals(data[i], data1[i])

    def testInvalidDftLength(self):
        caught = False
        try:
            pa_tools = StftManager(dft_length=-1)
        except ValueError:
            caught = True
        self.assertEquals(caught, True)

    def testInvalidWindowLength(self):
        caught = False
        try:
            pa_tools = StftManager(window_length=-1)
        except ValueError:
            caught = True
        self.assertEquals(caught, True)

    def testInvalidNChannels(self):
        caught = False
        try:
            pa_tools = StftManager(n_channels=-1)
        except ValueError:
            caught = True
        self.assertEquals(caught, True)

    def testInvalidHopLength(self):
        caught = False
        try:
            pa_tools = StftManager(hop_length=-1)
        except ValueError:
            caught = True
        self.assertEquals(caught, True)

    def testHopGreaterThanWindow(self):
        caught = False
        try:
            pa_tools = StftManager(hop_length=1024, window_length=512)
        except ValueError:
            caught = True
        self.assertEquals(caught, True)

    def testGetDFT(self):
        reals, imags = self.stft_dft.getDFTs()
        print reals
        print imags
