__author__ = 'adamjmiller'
import struct
import pyaudio
import threading
import numpy as np


class AudioBuffer:
    """
    A FIFO buffer class for audio data. Data is put in and
    retrieved as byte arrays (strings)

    NOTE: This class uses np.float32 data type to work properly
    with portaudio
    """
    _format = {
        pyaudio.paFloat32: 'f',
        pyaudio.paInt32: 'i',
        pyaudio.paInt16: 'h',
        pyaudio.paInt8: 'b',
        pyaudio.paUInt8: 'B'
    }

    def __init__(self, length, n_channels=1):
        """
        :param length: length of the buffer in samples
        :param n_channels: Number of channels present in the audio samples.
        """
        self._n_channels = n_channels
        self._length = length * self._n_channels  # Length in samples
        self._sample_size = pyaudio.get_sample_size(pyaudio.paFloat32)
        self._sample_format = self._format[pyaudio.paFloat32]
        # Intialize state variables
        self._size = 0
        self._write_start = 0
        self._read_start = 0
        # Instantiate buffer
        self._buffer = np.zeros(self._length * self._n_channels, dtype=np.float32)

    def write_samples(self, data):
        """
        Write data to the buffer using a numeric data format (as opposed
        to bytearray as used in write_bytes()
        :param data: a numpy array of data frames in the format specified when
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
        :return: list of samples in the format the was specified when
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
        Write data to the buffer, where the data is in the form of a
        bytearray
        :param data: Data to be written. Should be a bytearray (string)
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

        :param n_samples: number of frames of data to retrieve
        :return: 'n_bytes' bytes from the buffer in the form of a string
        """
        data = self.read_samples(n_samples)
        return struct.pack("%d%s" % (len(data), self._sample_format), *data)

    def reduce_channels(self, data, n_chan_in, n_chan_out):
        """
        Return array of samples consisting only of samples from
        the first 'n_chan_out' channels of the given data samples.
        Note that data should be in the form of data samples of float32
        type. It should not be a bytearray
        :param data: Data array of samples in the format returned by
                        read_samples()
        :param n_chan_out: Number of channels to retain in the modified
                            data array
        :return: Data from the first n_chan_out channels of the input data
        """
        new_size = len(data) * (float(n_chan_out) / float(n_chan_in))
        data_out = np.empty((new_size,))
        for n in range(len(data) / n_chan_in):
            data_out[n * n_chan_out: (n + 1) * n_chan_out] = \
                data[n * n_chan_in: n * n_chan_in + n_chan_out]
        return data_out

    def get_available_write(self):
        """
        :return: the amount of available space in the buffer in number
         of samples
        """
        return (self._length - self._size) / self._n_channels

    def get_available_read(self):
        """
        :return: the amount of data in the buffer that can be read in
        number of samples
        """
        return self._size / self._n_channels

    def _setup_events(self):
        """
        Setup the necessary threading.Event objects for synchronizing
        this buffer
        """
        new_audio_event = threading.Event()
        new_samples_event = threading.Event()

