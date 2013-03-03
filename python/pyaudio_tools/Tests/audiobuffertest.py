from pa_tools.audiobuffer import AudioBuffer

__author__ = 'adamjmiller'
import unittest
import pyaudio
import numpy.random

class AudioBufferTest(unittest.TestCase):

    def setUp(self):
        self.n_channels = 2
        self.buff32 = AudioBuffer(16, pyaudio.paFloat32, self.n_channels)
        self.buf = [
            AudioBuffer(16, pyaudio.paFloat32, self.n_channels),
            AudioBuffer(16, pyaudio.paInt32, self.n_channels),
            AudioBuffer(16, pyaudio.paInt16, self.n_channels),
            AudioBuffer(16, pyaudio.paInt8, self.n_channels),
            AudioBuffer(16, pyaudio.paUInt8, self.n_channels)]
        #self.buf = AudioBuffer(16, pyaudio.paInt32)
        self.data_str = '\x01\x02\x03\x04'
        self.data_strs = [
            '\x01\x02\x03\x04',
            '\x01\x02\x03\x04',
            '\x01\x02',
            '\x01',
            '\x01']
        self.num_types = 5
        self.data12 = [''] * self.num_types
        for i in range(len(self.data_strs)):
            self.data12[i] = self.data_str * self.n_channels * 12

    def data_of_length(self, length):
        return self.data_str * self.n_channels * length


    def testBufWrite(self):
        self.buff32.write_bytes(self.data_of_length(12))
        self.assertEquals(self.buff32.get_available_read(), 12)
        self.buff32.write_bytes(self.data_of_length(4))
        self.assertEquals(self.buff32.get_available_read(), 16)

    def testOverwrite(self):
        self.buff32.write_bytes(self.data_of_length(12))
        data6 = self.data_of_length(6)
        caught = False
        try:
            self.buff32.write_bytes(data6)
        except ValueError as e:
            print e.message
            caught = True
        self.assertEquals(caught, True)

    def testOverRead(self):
        data4 = self.data_of_length(4)
        self.buff32.write_bytes(data4)
        data_read = self.buff32.read_bytes(12)
        self.assertEquals(data_read, data4)

    def testWriteSamples(self):
        data = list(numpy.random.rand(12 * self.n_channels))
        data4 = list(numpy.random.rand(4 * self.n_channels))
        self.buff32.write_samples(data)
        self.assertEquals(self.buff32.get_available_read(), 12)
        self.buff32.write_samples(data4)
        self.assertEquals(self.buff32.get_available_read(), 16)

    def testReadSamples(self):
        data = list(numpy.random.rand(12 * self.n_channels))
        data4 = list(numpy.random.rand(4 * self.n_channels))
        self.buff32.write_samples(data)
        self.assertEquals(self.buff32.get_available_read(), 12)
        data_read = self.buff32.read_samples(12)
        self.assertListFloatEqual(data, data_read)
        self.buff32.write_samples(data4)
        self.assertEquals(self.buff32.get_available_read(), 4)
        data_read = self.buff32.read_samples(4)
        self.assertListFloatEqual(data4, data_read)

    def testWriteArray(self):
        data = numpy.random.rand(12 * self.n_channels)
        data4 = list(numpy.random.rand(4 * self.n_channels))
        self.buff32.write_samples(data)
        self.assertEquals(self.buff32.get_available_read(), 12)
        self.buff32.write_samples(data4)
        self.assertEquals(self.buff32.get_available_read(), 16)

    def testReadArray(self):
        data = numpy.random.rand(12 * self.n_channels)
        data4 = numpy.random.rand(4 * self.n_channels)
        self.buff32.write_samples(data)
        self.assertEquals(self.buff32.get_available_read(), 12)
        data_read = self.buff32.read_samples(12)
        self.assertListFloatEqual(data, data_read)
        self.buff32.write_samples(data4)
        self.assertEquals(self.buff32.get_available_read(), 4)
        data_read = self.buff32.read_samples(4)
        self.assertListFloatEqual(data_read, data4)

    def testRead(self):
        self.buff32.write_bytes(self.data_of_length(12))
        read_data = self.buff32.read_bytes(12)
        print read_data
        self.assertEquals(read_data, self.data_of_length(12))

    def testWraparoundWrite(self):
        self.buff32.write_bytes(self.data_of_length(12))
        self.buff32.read_bytes(12)
        self.buff32.write_bytes(self.data_of_length(12))
        self.assertEquals(self.buff32.get_available_read(), 12)

    def testWraparoundRead(self):
        self.buff32.write_bytes(self.data_of_length(12))
        self.buff32.read_bytes(12)
        self.buff32.write_bytes(self.data_of_length(12))
        self.assertEquals(self.buff32.get_available_read(), 12)
        data_read = self.buff32.read_bytes(12)
        self.assertEquals(data_read, self.data_of_length(12))
        self.assertEquals(self.buff32.get_available_read(), 0)


    def testInvalidData(self):
        caught = False
        try:
            self.buff32.write_bytes([2]*12)
        except ValueError as e:
            print e.message
            caught = True
        self.assertEquals(caught, True)

    def testMultipleWraps(self):
        self.buff32.write_bytes(self.data_of_length(12))
        self.buff32.read_bytes(8)
        self.buff32.write_bytes(self.data_of_length(12))
        self.buff32.read_bytes(16)
        self.buff32.write_bytes(self.data_of_length(12))
        self.buff32.read_bytes(9)
        self.buff32.write_bytes(self.data_of_length(12))
        self.buff32.read_bytes(11)
        self.buff32.write_bytes(self.data_of_length(12))
        data_read = self.buff32.read_bytes(12)
        self.assertEquals(self.buff32.get_available_read(), 4)
        self.assertEquals(data_read, self.data_of_length(12))

    def assertListFloatEqual(self, list1, list2):
        if not len(list1) == len(list2):
            raise AssertionError("Lists differ in lenght. Cannot be equal")
        for i in range(len(list1)):
            self.assertLessEqual(abs(list1[i] - list2[i]), 1e-4)


    def tearDown(self):
        print "Completed"

if __name__ == '__main__':
    unittest.main()