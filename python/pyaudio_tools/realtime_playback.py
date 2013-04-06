__author__ = 'adamjmiller'
import pyaudio
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from pa_tools.audiohelper import AudioHelper
from pa_tools.audiobuffer import AudioBuffer
from pa_tools.stftmanager import StftManager
from dft_conversion import *


# Setup constants
SAMPLE_TYPE = pyaudio.paFloat32
DATA_TYPE = np.float32
SAMPLE_SIZE = pyaudio.get_sample_size(SAMPLE_TYPE)
SAMPLE_RATE = 16000
FRAMES_PER_BUF = 1024  # Do not go below 64, or above 2048
FFT_LENGTH = FRAMES_PER_BUF
WINDOW_LENGTH = FFT_LENGTH
HOP_LENGTH = WINDOW_LENGTH / 2
NUM_CHANNELS_IN = 2
NUM_CHANNELS_OUT = 2
DO_PLOT = True
PLOT_FREQ = 1  # For PLOT_FREQ = n, will plot every n loops
TIMEOUT = 2  # Number of seconds to wait for new samples before giving up

# Track whether we have quit or not
done = False

# Setup data buffers
in_buf = AudioBuffer(length=4 * FRAMES_PER_BUF, n_channels=NUM_CHANNELS_IN)
out_buf = AudioBuffer(length=4 * FRAMES_PER_BUF, n_channels=NUM_CHANNELS_OUT)


def read_in_data(in_data, frame_count, time_info, status_flags):
    if done:  # Must do this or calls to stop_stream may not succeed
        return None, pyaudio.paComplete
    write_num = in_buf.get_available_write()
    if write_num > frame_count:
        write_num = frame_count
    in_buf.write_bytes(in_data[:(write_num * SAMPLE_SIZE * NUM_CHANNELS_IN)])
    in_buf.notify_of_audio()
    return None, pyaudio.paContinue


def write_out_data(in_data, frame_count, time_info, status_flags):
    if done:  # Must do this or calls to stop_stream may not succeed
        return None, pyaudio.paComplete
    if out_buf.get_available_read() >= frame_count:
        return out_buf.read_bytes(frame_count), pyaudio.paContinue
    else:  # Return empty data (returning None will trigger paComplete)
        return '\x00' * frame_count * SAMPLE_SIZE * NUM_CHANNELS_IN, pyaudio.paContinue


def process_dfts(dfts):
    for (reals, imags) in dfts:
        for real in reals:
            process_dft_buf(real)
        for imag in imags:
            process_dft_buf(imag)


def process_dft_buf(buf):
    # Low pass filter:
    for i in range(len(buf)):
        if i > FFT_LENGTH / 18:
            buf[i] = 0
    pass


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
    stft = StftManager(dft_length=FFT_LENGTH,
                       window_length=WINDOW_LENGTH,
                       hop_length=HOP_LENGTH,
                       use_window_fcn=True,
                       n_channels=NUM_CHANNELS_IN,
                       dtype=DATA_TYPE)

    # Get devices
    in_device = helper.get_input_device_from_user()
    out_device = helper.get_output_device_from_user()

    # Setup streams
    in_stream = pa.open(rate=SAMPLE_RATE,
                        channels=NUM_CHANNELS_IN,
                        format=SAMPLE_TYPE,
                        frames_per_buffer=FRAMES_PER_BUF,
                        input=True,
                        input_device_index=int(in_device['index']),
                        stream_callback=read_in_data)
    out_stream = pa.open(rate=SAMPLE_RATE,
                         channels=NUM_CHANNELS_OUT,
                         format=SAMPLE_TYPE,
                         output=True,
                         frames_per_buffer=FRAMES_PER_BUF,
                         output_device_index=int(out_device['index']),
                         stream_callback=write_out_data)

    # Start recording/playing back
    in_stream.start_stream()
    out_stream.start_stream()

    # Start thread to check for user quit
    quit_thread = threading.Thread(target=check_for_quit)
    quit_thread.start()

    # Setup plotting
    if DO_PLOT:
        plt.ion()
        fig = plt.figure()
        # Setup time plot and bounds. Each loop only update data
        time_ax = fig.add_subplot(211)
        time_plot, = time_ax.plot(np.arange(WINDOW_LENGTH), np.zeros(WINDOW_LENGTH))
        time_ax.set_ylim(-.5, .5)
        time_ax.set_xlim(0, WINDOW_LENGTH)
        time_ax.set_ylabel('Amplitude')
        time_ax.set_xlabel('Sample')
        # Setup frequency plot and bounds. Each loop only update data
        freq_ax = fig.add_subplot(212)
        freq_plot, = freq_ax.plot(np.linspace(0, SAMPLE_RATE / 2., FFT_LENGTH / 2 + 1), np.zeros(FFT_LENGTH / 2 + 1))
        freq_ax.set_ylim(0, 50)
        freq_ax.set_ylabel('Magnitude')
        freq_ax.set_xlabel('Frequency (Hz)')
        # Show figure
        plt.show(block=False)

    data1 = np.zeros(WINDOW_LENGTH, dtype=DATA_TYPE)
    count = 0
    try:
        while in_stream.is_active() or out_stream.is_active():
            data_is_available = in_buf.wait_for_read(WINDOW_LENGTH, TIMEOUT)
            if data_is_available:# and out_buf.get_available_write() >= WINDOW_LENGTH:
            #available = min(in_buf.get_available_read(), out_buf.get_available_write())
            #if available >= WINDOW_LENGTH:  # If enough space to transfer data
                # Get data from the circular buffer
                data = in_buf.read_samples(WINDOW_LENGTH)
                # Perform an stft
                stft.performStft(data)
                # Process dfts from windowed segments of input
                dfts = stft.getDFTs()
                process_dfts(dfts)
                if DO_PLOT:
                    # Must update here because dfts are altered upon calling ISTFT since
                    # the dft is performed in place
                    fft = to_full_fft(dfts[0][0][0], dfts[0][1][0])
                # Get the istft of the processed data
                new_data = stft.performIStft()
                # Alter data so it can be written out
                if NUM_CHANNELS_IN != NUM_CHANNELS_OUT:
                    new_data = out_buf.reduce_channels(new_data, NUM_CHANNELS_IN, NUM_CHANNELS_OUT)
                print out_buf.get_available_write()
                if out_buf.get_available_write() >= WINDOW_LENGTH:
                    out_buf.write_samples(new_data)
                # Take care of plotting
                if DO_PLOT:
                    # Time plot
                    if NUM_CHANNELS_OUT != 1:
                        plot_data = out_buf.reduce_channels(new_data, NUM_CHANNELS_OUT, 1)
                    else:
                        plot_data = new_data
                    time_plot.set_ydata(plot_data)
                    # Frequency plot
                    freq_plot.set_ydata(np.abs(fft[:FFT_LENGTH / 2 + 1]))
                    # Update plot
                    fig.canvas.draw()
            #time.sleep(.001)
    except KeyboardInterrupt:
        print "Program interrupted"
        done = True

    # Clean up
    print "Cleaning up"
    in_stream.stop_stream()
    in_stream.close()
    out_stream.stop_stream()
    out_stream.close()
    pa.terminate()
    print "Done"


