__author__ = 'adamjmiller'
import pyaudio
import wave
import struct
import numpy as np
import threading
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pa_tools.constants as consts
import pa_tools.mattools as mat
from pa_tools.audiohelper import AudioHelper
from pa_tools.audiobuffer import AudioBuffer
from pa_tools.stftmanager import StftManager
from pa_tools.distributionlocalizer import DistributionLocalizer
from pa_tools.beamformer import BeamFormer

# Setup constants
SAMPLE_TYPE = pyaudio.paFloat32
DATA_TYPE = np.float32
SAMPLE_SIZE = pyaudio.get_sample_size(SAMPLE_TYPE)
SAMPLE_RATE = 44100
FRAMES_PER_BUF = 4096  # For 44100 Fs, be careful going over 4096, loud sounds may occur...
FFT_LENGTH = FRAMES_PER_BUF
WINDOW_LENGTH = FFT_LENGTH
HOP_LENGTH = WINDOW_LENGTH / 2
NUM_CHANNELS_IN = 7
NUM_CHANNELS_OUT = 1
N_THETA = 30
N_PHI = N_THETA * 1 / 2 # 3 / 4
PLOT_CARTES = False
PLOT_POLAR = True
EXTERNAL_PLOT = False
PLAY_AUDIO = True
DO_BEAMFORM = True
RECORD_AUDIO = False
OUTFILE_NAME = 'nonbeamformed.wav'
TIMEOUT = 1
# Setup mics
R = 0.0375
H = 0.07
mic_layout = np.array([[0, 0, H],
                       [R, 0, 0],
                       [R*math.cos(math.pi/3), R*math.sin(math.pi/3), 0],
                       [-R*math.cos(math.pi/3), R*math.sin(math.pi/3), 0],
                       [-R, 0, 0],
                       [-R*math.cos(math.pi/3), -R*math.sin(math.pi/3), 0],
                       [R*math.cos(math.pi/3), -R*math.sin(math.pi/3), 0]])
# Track whether we have quit or not
done = False
switch_beamforming = False  # Switch beamforming from on to off or off to on

# Events for signaling new data is available
audio_produced_event = threading.Event()
data_produced_event = threading.Event()

# Setup data buffers - use 4 * buffer length in case data get's backed up
# at any point, so it will not be lost
in_buf = AudioBuffer(length=4 * FRAMES_PER_BUF, n_channels=NUM_CHANNELS_IN)
out_buf = AudioBuffer(length=4 * FRAMES_PER_BUF, n_channels=NUM_CHANNELS_OUT)

# Setup record buffer
N_SECS_RECORD = 20
N_RECORD_FRAMES = N_SECS_RECORD * SAMPLE_RATE
record_buf = AudioBuffer(length=N_RECORD_FRAMES, n_channels=NUM_CHANNELS_OUT)


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
        return '\x00' * frame_count * SAMPLE_SIZE * NUM_CHANNELS_OUT, pyaudio.paContinue


def process_dfts(dfts):
    for (reals, imags) in dfts:
        for real in reals:
            process_dft_buf(real)
        for imag in imags:
            process_dft_buf(imag)


def process_dft_buf(buf):
    # Low pass filter:
    for i in range(len(buf)):
        if i > FFT_LENGTH / 16:
            buf[i] = 0
    pass


def check_for_quit():
    global done
    global switch_beamforming
    while True:
        read_in = raw_input()
        if read_in == "q":
            print "User has chosen to quit."
            done = True
            break
        if read_in == "b":
            switch_beamforming = True


def print_dfts(dfts):
    print "Printing DFTS:"
    print dfts
    sample_len = 12
    for k in range(len(dfts)):
        print "Channel %d" % k
        reals = dfts[k][0]
        imags = dfts[k][1]
        for i in range(len(reals)):
            print "Reals %d:" % i
            out_str = ""
            for j in range(sample_len):
                out_str += "%f\t" % reals[i][j]
            print out_str
        for i in range(len(imags)):
            print "Imags %d:" % i
            out_str = ""
            for j in range(sample_len):
                out_str += "%f\t" % reals[i][j]
            print out_str


def make_wav():
    SHORT_MAX = (2 ** 15) - 1
    data = record_buf.read_whole_buffer()
    sample_width = 2  # Bytes
    params = (NUM_CHANNELS_OUT, sample_width, SAMPLE_RATE, N_RECORD_FRAMES, 'NONE', 'not compressed')
    outwav = wave.open(OUTFILE_NAME, 'w')
    outwav.setparams(params)

    # Convert to shorts
    data = np.asarray(data * .5 * SHORT_MAX, dtype=np.int16)
    data_bytes = struct.pack('%dh' % NUM_CHANNELS_OUT * N_RECORD_FRAMES, *data)

    # Make plot
    plt.plot(data[2 * SAMPLE_RATE:2.6 * SAMPLE_RATE]) # Plot 1 sec of data
    if DO_BEAMFORM:
        plt.savefig('plotbeamformed.png')
    else:
        plt.savefig('plotnonbeamformed.png')
    plt.show()

    # Write out to file
    outwav.writeframes(data_bytes)
    outwav.close()


def localize():
    global switch_beamforming
    global DO_BEAMFORM
    # Setup pyaudio instances
    pa = pyaudio.PyAudio()
    helper = AudioHelper(pa)
    localizer = DistributionLocalizer(mic_positions=mic_layout,
                                      dft_len=FFT_LENGTH,
                                      sample_rate=SAMPLE_RATE,
                                      n_theta=N_THETA,
                                      n_phi=N_PHI)
    beamformer = BeamFormer(mic_layout, SAMPLE_RATE)

    # Setup STFT object
    stft = StftManager(dft_length=FFT_LENGTH,
                       window_length=WINDOW_LENGTH,
                       hop_length=HOP_LENGTH,
                       use_window_fcn=True,
                       n_channels=NUM_CHANNELS_IN,
                       dtype=DATA_TYPE)

    # Setup devices
    in_device = helper.get_input_device_from_user()
    if PLAY_AUDIO:
        out_device = helper.get_output_device_from_user()
    else:
        out_device = helper.get_default_output_device_info()

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

    # Setup directions and alignment matrices
    direcs = localizer.get_directions()
    align_mats = localizer.get_pos_align_mat()

    # Plotting setup
    if PLOT_CARTES:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.show(block=False)
        x = localizer.to_spher_grid(direcs[0, :])
        y = localizer.to_spher_grid(direcs[1, :])
        z = localizer.to_spher_grid(direcs[2, :])
        #scat = ax.scatter(x, y, z, s=100)
    if PLOT_POLAR:
        fig = plt.figure()
        ax = fig.add_axes([.1, .1, .8, .8], projection='polar')
        ax.set_rlim(0, 1)
        plt.show(block=False)
        # Setup space for plotting in new coordinates
        spher_coords = localizer.get_spher_directions()
        pol = localizer.to_spher_grid(spher_coords[2, :])
        weight = 1. - .3 * np.sin(2 * pol)  # Used to pull visualization off edges
        r = np.sin(pol) * weight
        theta = localizer.to_spher_grid(spher_coords[1, :])
    if EXTERNAL_PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.show(block=False)

    count = 0
    try:
        global done
        while in_stream.is_active() or out_stream.is_active():
            data_available = in_buf.wait_for_read(WINDOW_LENGTH, TIMEOUT)
            if data_available:
                if switch_beamforming:
                    DO_BEAMFORM = not DO_BEAMFORM
                    switch_beamforming = False
                # Get data from the circular buffer
                data = in_buf.read_samples(WINDOW_LENGTH)
                # Perform an stft
                stft.performStft(data)
                # Process dfts from windowed segments of input
                dfts = stft.getDFTs()
                rffts = mat.to_all_real_matlab_format(dfts)
                d = localizer.get_distribution_real(rffts[:, :, 0])
                ind = np.argmax(d)
                u = 1.5 * direcs[:, ind]  # Direction of arrival

                # Do beam forming
                if DO_BEAMFORM:
                    align_mat = align_mats[:, :, ind]
                    filtered = beamformer.filter_real(rffts, align_mat)
                    mat.set_dfts_real(dfts, filtered, n_channels=2)

                    # Get beam plot
                    freq = 6000.  # Hz
                    response = beamformer.get_beam(align_mat, align_mats, freq)
                    response = localizer.to_spher_grid(response)

                # Take car of plotting
                if count % 1 == 0:
                    if PLOT_CARTES:
                        ax.cla()
                        ax.grid(False)
                        d = localizer.to_spher_grid(d / (np.max(d) + consts.EPS))
                        ax.scatter(x, y, z, c=d, s=40)
                        #ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolor=plt.cm.gist_heat(d))
                        ax.plot([0, u[0]], [0, u[1]], [0, u[2]], c='black', linewidth=3)
                        if DO_BEAMFORM:
                            X = response * x
                            Y = response * y
                            Z = response * z
                            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='white')
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.set_zlim(0, 1)
                        #ax.view_init(90, -90)
                        fig.canvas.draw()
                    if PLOT_POLAR:
                        plt.cla()
                        d = localizer.to_spher_grid(d)
                        con = ax.contourf(theta, r, d, vmin=0, vmax=40)
                        con.set_cmap('gist_heat')
                        if DO_BEAMFORM:
                            response = response[-1, :]  # Pick which polar angle sample to use
                            ax.plot(theta[0, :], response, 'cyan', linewidth=4)
                            ax.set_rlim(0, 1)
                        fig.canvas.draw()
                count += 1

                # Get the istft of the processed data
                if PLAY_AUDIO or RECORD_AUDIO:
                    new_data = stft.performIStft()
                    new_data = out_buf.reduce_channels(new_data, NUM_CHANNELS_IN, NUM_CHANNELS_OUT)
                    # Write out the new, altered data
                    if PLAY_AUDIO:
                        if out_buf.get_available_write() >= WINDOW_LENGTH:
                            out_buf.write_samples(new_data)
                    if RECORD_AUDIO:
                        if record_buf.get_available_write() >= WINDOW_LENGTH:
                            record_buf.write_samples(new_data)


    except KeyboardInterrupt:
        print "Program interrupted"
        done = True


    print "Cleaning up"
    in_stream.stop_stream()
    in_stream.close()
    out_stream.stop_stream()
    out_stream.close()
    pa.terminate()

    # Take care of output file
    if RECORD_AUDIO:
        print "Writing output file"
        make_wav()

    print "Done"

if __name__ == '__main__':
    localize()
