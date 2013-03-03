__author__ = 'adamjmiller'
import struct
import pyaudio
import numpy as np


class AudioBuffer:
    """
    A FIFO buffer class for audio data. Data is put in and
    retrieved as byte arrays (strings)

    NOTE: This class will not work for custom sample formats or
    paInt24 samples formats

    @param length: length of the buffer in samples
    @param d_type: sample format type. Should be in pyaudio form (paFloat32 e.g.)
    """
    _format = {
        pyaudio.paFloat32: 'f',
        pyaudio.paInt32: 'i',
        pyaudio.paInt16: 'h',
        pyaudio.paInt8: 'b',
        pyaudio.paUInt8: 'B'
    }

    def __init__(self, length=2048, d_type=pyaudio.paFloat32, n_channels=1):
        self._n_channels = n_channels
        self._length = length * self._n_channels # Length in samples
        self._sample_size = pyaudio.get_sample_size(d_type)
        self._sample_format = self._format[d_type]
        self._size = 0
        self._write_start = 0
        self._read_start = 0
        self._buffer = np.zeros(self._length * self._n_channels, dtype=np.float32)

    def write_samples(self, data):
        """
        Write data to the buffer
        @param data: a numpy array of data frames in the format specified when
                     creating the AudioBuffer. Data for different channels
                     should be interlaced
        """
        try:
            iter(data)
        except:
            raise ValueError("Input must be an iterable collection of data in correct sammple format")
            # Deal with wraparound in buffer
        if len(data) + self._write_start > self._length:
            num_to_end = self._length - self._write_start
            self._buffer[self._write_start:self._length] = data[:num_to_end]
            self._buffer[:(self._write_start + len(data)) % self._length] = data[num_to_end:]
        else:  # No wraparound
            self._buffer[self._write_start:self._write_start + len(data)] = data
            # Update state variables
        self._write_start = (self._write_start + len(data)) % self._length
        self._size += len(data)

    def read_samples(self, n_samples):
        """
        Get data from buffer in the form of samples.
        @return: list of samples in the format the was specified when
                    creating the AudioBuffer
        """
        n_samples = n_samples * self._n_channels
        if n_samples > self._size:
            n_samples = self._size
        data = np.empty(n_samples, dtype=np.float32)
        # Check for wraparound
        if n_samples + self._read_start > self._length:
            n_before = self._length - self._read_start
            n_after = (self._read_start + n_samples) % self._length
            data[:n_before] = self._buffer[self._read_start:self._length]
            data[n_before:] = self._buffer[:n_after]
        else:
            data[:] = self._buffer[self._read_start:self._read_start + n_samples]
        self._size -= n_samples
        self._read_start = (self._read_start + n_samples) % self._length
        return data

    def write_bytes(self, data):
        """
        Pushes the input data to the end of the buffer
        """
        # Ensure input is of proper type and size
        if not type(data) == str:
            raise ValueError("Input data should be a bytearray (string)")
        if len(data) / self._sample_size > self._length - self._size:
            raise ValueError("Input size larger than available space in buffer")
            # Get list representation
        data_list = np.array((struct.unpack(
            "%d%s" % (len(data) / self._sample_size, self._sample_format), data)), dtype=np.float32)
        self.write_samples(data_list)

    def read_bytes(self, n_samples):
        """
        returns 'n_bytes' bytes of data in the form of a string
        If there are less than n_bytes available, it will return all
        data in the buffer

        @param n_samples: number of frames of data to retrieve
        @return: 'n_bytes' bytes from the buffer in the form of a string
        """
        data = self.read_samples(n_samples)
        return struct.pack("%d%s" % (len(data), self._sample_format), *data)

    def get_available_write(self):
        """
        @return: the amount of available space in the buffer in number
         of samples
        """
        return (self._length - self._size) / self._n_channels

    def get_available_read(self):
        """
        @return: the amount of data in the buffer that can be read in
        number of samples
        """
        return self._size / self._n_channels

