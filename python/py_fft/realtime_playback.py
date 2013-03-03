__author__ = 'adamjmiller'
import pyaudio
import time
import numpy as np
import threading
from pa_tools.audiohelper import AudioHelper
from pa_tools.audiobuffer import AudioBuffer
from pa_tools.stftmanager import StftManager


# Setup constants
SAMPLE_TYPE = pyaudio.paFloat32
DATA_TYPE = np.float32
SAMPLE_SIZE = pyaudio.get_sample_size(SAMPLE_TYPE)
SAMPLE_RATE = 44100
FFT_LENGTH = 512
WINDOW_LENGTH = FFT_LENGTH
FRAMES_PER_BUF= 512
HOP_LENGTH = 256
NUM_CHANNELS = 1

# Track whether we have quit or not
done = False

# Setup data buffers
in_buf = AudioBuffer(n_channels=NUM_CHANNELS)
out_buf = AudioBuffer(n_channels=NUM_CHANNELS)


def read_in_data(in_data, frame_count, time_info, status_flags):
    if done:  # Must do this or calls to stop_stream may not succeed
        return None, pyaudio.paComplete
    write_num = in_buf.get_available_write()
    if write_num > frame_count:
        write_num = frame_count
    in_buf.write_bytes(in_data[:(write_num * SAMPLE_SIZE * NUM_CHANNELS)])
    return None, pyaudio.paContinue


def write_out_data(in_data, frame_count, time_info, status_flags):
    if done:  # Must do this or calls to stop_stream may not succeed
        return None, pyaudio.paComplete
    if out_buf.get_available_read() >= frame_count:
        return out_buf.read_bytes(frame_count), pyaudio.paContinue
    else:  # Return empty data (returning None will trigger paComplete)
        return '\x00' * frame_count * SAMPLE_SIZE * NUM_CHANNELS, pyaudio.paContinue


def process_dfts(reals, imags):
    for real in reals:
        process_dft_buf(real)
    for imag in imags:
        process_dft_buf(imag)


def process_dft_buf(buf):
    pass
    # Low pass filter:
    #for i in range(len(buf)):
    #    if i > FFT_LENGTH / 8:
    #        buf[i] = 0


def check_for_quit():
    global done
    while True:
        read_in = raw_input()
        if read_in == "q":
            done = True
            break


if __name__ == '__main__':
    # Setup pyaudio instances
    pa = pyaudio.PyAudio()
    helper = AudioHelper(pa)

    # Setup STFT object
    stft = StftManager(dft_length=FFT_LENGTH, window_length=WINDOW_LENGTH, hop_length=HOP_LENGTH, dtype=DATA_TYPE)

    # Get devices
    in_device = helper.get_input_device_from_user()
    out_device = helper.get_default_output_device_info()

    # Setup streams
    in_stream = pa.open(rate=SAMPLE_RATE,
                        channels=NUM_CHANNELS,
                        format=SAMPLE_TYPE,
                        frames_per_buffer=FRAMES_PER_BUF,
                        input=True,
                        input_device_index=int(in_device['index']),
                        stream_callback=read_in_data)
    out_stream = pa.open(rate=SAMPLE_RATE,
                         channels=NUM_CHANNELS,
                         format=SAMPLE_TYPE,
                         output=True,
                         frames_per_buffer=FRAMES_PER_BUF,
                         output_device_index=int(out_device['index']),
                         stream_callback=write_out_data)

    # Start recording/playing back
    in_stream.start_stream()
    out_stream.start_stream()

    # Start plotting thread
    quit_thread = threading.Thread(target=check_for_quit)
    quit_thread.start()

    data1 = np.zeros(WINDOW_LENGTH, dtype=DATA_TYPE)
    try:
        while in_stream.is_active() or out_stream.is_active():
            available = min(in_buf.get_available_read(), out_buf.get_available_write())
            if available >= WINDOW_LENGTH:  # If enough space to transfer data
                data = in_buf.read_samples(WINDOW_LENGTH)
                # Do STFT and ISTFT
                stft.performStft(data)
                reals, imags = stft.getDFTs()
                process_dfts(reals, imags)
                stft.performIStft(data1)
                out_buf.write_samples(data1)
            time.sleep(.001)
    except KeyboardInterrupt:
        print "Program interrupted"
        done = True

    print "Cleaning up"
    in_stream.stop_stream()
    in_stream.close()
    out_stream.stop_stream()
    out_stream.close()
    pa.terminate()
    print "Done"


