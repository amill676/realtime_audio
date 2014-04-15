__author__ = 'Adam Miller'
import pyaudio
import numpy as np
import threading
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pa_tools.audiohelper import AudioHelper
from pa_tools.audiobuffer import AudioBuffer
from pa_tools.stftmanager import StftManager
from pa_tools.distributionlocalizer import DistributionLocalizer
from searchspace import SearchSpace
from searchspace import SourcePlane
from camera import SonyCamera
from camera.info import *

# Camera constants
DISTANCE_TO_TEACHER = 22
TEACHER_NORMAL = np.array([1, 0, 0])
TEACHER_OFFSET = np.array([DISTANCE_TO_TEACHER, 0, 0])
DISTANCE_TO_STUDENTS = 7
STUDENT_NORMAL = np.array([0, 0, 1])
STUDENT_OFFSET = np.array([0, 0, -DISTANCE_TO_STUDENTS])
#MIC_LOC = np.array([DISTANCE_TO_TEACHER - 4, 0, -DISTANCE_TO_STUDENTS])
MIC_LOC = np.array([DISTANCE_TO_TEACHER - 5, -4, -DISTANCE_TO_STUDENTS])
CAMERA_LOC = np.array([0, 0, 0])
TRACKING_FREQ = 4
DO_TRACK = True
PLOT_SPACE = False
URL = "172.22.11.130"

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
NUM_CHANNELS_OUT = 2
N_THETA = 20
N_PHI = N_THETA / 2
PLOT_CARTES = False
EXTERNAL_PLOT = False
PLAY_AUDIO = False
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

# Events for signaling new data is available
audio_produced_event = threading.Event()
data_produced_event = threading.Event()

# Setup data buffers - use 4 * buffer length in case data get's backed up
# at any point, so it will not be lost
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
    while True:
        read_in = raw_input()
        if read_in == "q":
            print "User has chosen to quit."
            done = True
            break


def print_dfts(dfts):
    print "Printing DFTS:"
    print dfts
    sample_len = 12
    for k in range(len(dfts)):
        print "Channel %d" %k
        reals = dfts[k][0]
        imags = dfts[k][1]
        for i in range(len(reals)):
            print "Reals %d:" %i
            out_str = ""
            for j in range(sample_len):
                out_str += "%f\t" %reals[i][j]
            print out_str
        for i in range(len(imags)):
            print "Imags %d:" %i
            out_str = ""
            for j in range(sample_len):
                out_str += "%f\t" %reals[i][j]
            print out_str


def localize():

    # Setup search space
    # x vector points to front of class, -z vector points to floor
    teacher_plane = SourcePlane(TEACHER_NORMAL, TEACHER_OFFSET)
    student_plane = SourcePlane(STUDENT_NORMAL, STUDENT_OFFSET)
    space = SearchSpace(MIC_LOC, CAMERA_LOC, [teacher_plane, student_plane])

    # Setup camera
    forward = np.array([1, 0, 0])
    above = np.array([0, 0, 1])
    camera = SonyCamera(URL, forward, above)


    # Setup pyaudio instances
    pa = pyaudio.PyAudio()
    helper = AudioHelper(pa)
    localizer = DistributionLocalizer(mic_positions=mic_layout,
                                      dft_len=FFT_LENGTH,
                                      sample_rate=SAMPLE_RATE,
                                      n_theta=N_THETA,
                                      n_phi=N_PHI)

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

    # Plotting setup
    if PLOT_CARTES:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.show(block=False)
        scat = []
    if PLOT_SPACE:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Setup bounds
        xlo, xhi = (-5, DISTANCE_TO_TEACHER + 5)
        ylo, yhi = (-15, 15)
        zlo, zhi = (-15, 5)
        # Setup grid
        nx, ny = (200, 100)
        x = np.linspace(xlo, xhi, nx)
        y = np.linspace(ylo, yhi, ny)
        X, Y = np.meshgrid(x, y)
        n, m = (STUDENT_NORMAL, STUDENT_OFFSET)
        TP = (n.dot(m) - n[0] * X - n[1] * Y) / n[2] - 2
        # Plot markers for mic
        m = MIC_LOC
        ax.plot([MIC_LOC[0]], [MIC_LOC[1]], [MIC_LOC[2]], 'r.', markersize=10.)
        # Plot marker for camera
        c = CAMERA_LOC
        ax.plot([CAMERA_LOC[0]], [CAMERA_LOC[1]], [CAMERA_LOC[2]], 'b.', markersize=10.)
        # Draw lines from camera and mic to source
        source_loc = np.array([10, 0, 0])
        source_point, = ax.plot([source_loc[0]], [source_loc[1]], [source_loc[2]], 'black', marker='.', markersize=10.)
        s = source_loc
        camera_dir, = ax.plot([c[0], m[0]], [c[1], m[1]], [c[2], m[2]], 'blue')
        mic_dir, = ax.plot([m[0], m[0]], [m[1], m[1]], [m[2], m[2]], 'red')
        #ax.plot_surface(X, Y, TP)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_zlim(zlo, zhi)
        ax.view_init(elev=25, azim=-120)
        plt.show(block=False)
    if EXTERNAL_PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.show(block=False)

    count = 0
    prev_direc = np.array([0, 0, 0])
    direcs = localizer.get_directions()
    try:
        global done
        while in_stream.is_active() or out_stream.is_active():
            data_available = in_buf.wait_for_read(WINDOW_LENGTH, TIMEOUT)
            if data_available:
                # Get data from the circular buffer
                data = in_buf.read_samples(WINDOW_LENGTH)
                # Perform an stft
                stft.performStft(data)
                # Process dfts from windowed segments of input
                dfts = stft.getDFTs()
                d = localizer.get_3d_real_distribution(dfts)
                ind = np.argmax(d)
                u = 1.5 * direcs[:, ind]  # Direction of arrival

                if DO_TRACK and count % TRACKING_FREQ == 0:
                    #v = np.array([1, 0, 1])
                    v = u
                    direc = space.get_camera_dir(v)
                    if not direc.any():
                        direc = prev_direc
                    else:
                        prev_direc = direc
                    # Send camera new direction
                    camera.face_direction(direc)

                    if PLOT_SPACE:
                        if direc.any():
                            src = space.get_source_loc(u)
                            source_point.set_xdata([src[0]])
                            source_point.set_ydata([src[1]])
                            source_point.set_3d_properties(zs=[src[2]])
                        cam_src = CAMERA_LOC + 30 * direc
                        mic_src = MIC_LOC + 30 * u
                        # Update camera line
                        camera_dir.set_xdata([CAMERA_LOC[0], cam_src[0]])
                        camera_dir.set_ydata([CAMERA_LOC[1], cam_src[1]])
                        camera_dir.set_3d_properties(zs=[CAMERA_LOC[2], cam_src[2]])
                        # Update mic line
                        mic_dir.set_xdata([MIC_LOC[0], mic_src[0]])
                        mic_dir.set_ydata([MIC_LOC[1], mic_src[1]])
                        mic_dir.set_3d_properties(zs=[MIC_LOC[2], mic_src[2]])
                        plt.draw()

                # Take care of plotting
                if count % 1 == 0:
                    if PLOT_CARTES:
                        plt.cla()
                        ax.scatter(direcs[0, :], direcs[1, :], direcs[2, :], s=30, c=d[:])
                        ax.plot([0, u[0]], [0, u[1]], [0, u[2]], c='blue')
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.set_zlim(0, 1)
                        plt.draw()
                count += 1

                # Get the istft of the processed data
                if PLAY_AUDIO:
                    new_data = stft.performIStft()
                    new_data = out_buf.reduce_channels(new_data, NUM_CHANNELS_IN, NUM_CHANNELS_OUT)
                    # Write out the new, altered data
                    if out_buf.get_available_write() >= WINDOW_LENGTH:
                        out_buf.write_samples(new_data)
                        #time.sleep(.05)
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

if __name__ == '__main__':
    localize()
