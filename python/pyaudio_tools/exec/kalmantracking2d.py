import struct
import threading
import math
import cv2
import os.path

import pyaudio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pa_tools.constants as consts
import mattools.mattools as mat
import pa_tools.plottools as plotting
from pa_tools.audiohelper import AudioHelper
from pa_tools.audiobuffer import AudioBuffer
from pa_tools.stftmanager import StftManager
from pa_tools.kalmantrackinglocalizer import KalmanTrackingLocalizer
from pa_tools.gridtrackinglocalizer import GridTrackingLocalizer
from pa_tools.beamformer import BeamFormer
from searchspace import SearchSpace
from searchspace import OrientedSourcePlane


# Setup constants
SAMPLE_TYPE = pyaudio.paFloat32
DATA_TYPE = np.float32
SAMPLE_SIZE = pyaudio.get_sample_size(SAMPLE_TYPE)
SAMPLE_RATE = 44100
FRAMES_PER_BUF = 2048  # For 44100 Fs, be careful going over 4096, loud sounds may occur...
FFT_LENGTH = FRAMES_PER_BUF
WINDOW_LENGTH = FFT_LENGTH
HOP_LENGTH = WINDOW_LENGTH / 2
NUM_CHANNELS_IN = 4
NUM_CHANNELS_OUT = 1
N_THETA = 100
N_PHI = 1
PLOT_POLAR = False
PLOT_CARTES = True
PLOT_2D = False
EXTERNAL_PLOT = False
PLAY_AUDIO = False
DO_BEAMFORM = False
RECORD_AUDIO = False
VIDEO_OVERLAY = False
OUTFILE_NAME = 'nonbeamformed.wav'
FIG_OUTFILE_BASENAME = 'fig'
FIG_DIRECTORY_NAME = 'figures'
FIG_OUTFILE_NUMBER = 0  # Suffix for filename to avoid overwrites
TIMEOUT = 1
# Source planes and search space
SOURCE_PLANE_NORMAL = np.array([0, -1, 0])
SOURCE_PLANE_UP = np.array([0, 0 , 1])
SOURCE_PLANE_OFFSET = np.array([0, 4, 0])
SOURCE_LOCATION_COV = np.array([[6, 0], [0, .01]])
MIC_LOC = np.array([0, 0, 0])
CAMERA_LOC = np.array([0, 0, 0])
TIME_STEP = .1
STATE_TRANSITION_MAT = np.array([[1, 0, 0, TIME_STEP, 0, 0],
                                 [0, 1, 0, 0, TIME_STEP, 0],
                                 [0, 0, 1, 0, 0, TIME_STEP],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]])
#STATE_COV_MAT = 5 * np.identity(6, consts.REAL_DTYPE)
STATE_COV_MAT = np.array([[.01, 0, 0, 0, 0, 0],
                          [0, .01, 0, 0, 0, 0],
                          [0 ,0, .01, 0, 0, 0],
                          [0, 0, 0, .01, 0, 0],
                          [0, 0, 0, 0, .01, 0],
                          [0, 0, 0, 0, 0, .01]])
EMISSION_MAT = np.hstack((np.identity(3), np.zeros((3,3))))
EMISSION_COV = np.array([[90, 0, 0], [0, .1, 0], [0, 0, 1]], dtype=consts.REAL_DTYPE)
MIC_FORWARD = np.array([0, 1, 0])
MIC_ABOVE = np.array([0, 0, 1])

# Setup printing
np.set_printoptions(precision=2, suppress=True)
# Setup figure size
plotting.setup_halfpage_figsize()

# Setup mics
mic_layout = np.array([[.03, 0], [-.01, 0], [.01, 0], [-.03, 0]])
# Track whether we have quit or not
done = False
switch_beamforming = False  # Switch beamforming from on to off or off to on
save_fig = False  # For saving figure to file

# Events for signaling new data is available
audio_produced_event = threading.Event()
data_produced_event = threading.Event()

# Setup data buffers - use 4 * buffer length in case data get's backed up
# at any point, so it will not be lost
in_buf = AudioBuffer(length=4 * FRAMES_PER_BUF, n_channels=NUM_CHANNELS_IN)
out_buf = AudioBuffer(length=4 * FRAMES_PER_BUF, n_channels=NUM_CHANNELS_OUT)

# Setup record buffer
N_SECS_RECORD = 40
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
    global save_fig
    while True:
        read_in = raw_input()
        if read_in == "q":
            print "User has chosen to quit."
            done = True
            break
        if read_in == "b":
            switch_beamforming = True
        if read_in == "s":
            save_fig = True

def get_fig_name():
    global FIG_OUTFILE_NUMBER
    global FIG_DIRECTORY_NAME
    global FIG_OUTFILE_BASENAME
    if not os.path.exists('figures'):
        os.makedirs(FIG_DIRECTORY_NAME)
    filename = FIG_DIRECTORY_NAME + '/' + FIG_OUTFILE_BASENAME + \
            str(FIG_OUTFILE_NUMBER) + '.png'
    while os.path.exists(filename):
        FIG_OUTFILE_NUMBER += 1
        filename = FIG_DIRECTORY_NAME + '/' + FIG_OUTFILE_BASENAME + \
                str(FIG_OUTFILE_NUMBER) + '.png'
    return filename

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

def setup_2d_handle(ax, n_past_samples, est_color, title='', discard_edge_n = 0):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    sample_mat = np.zeros((N_THETA, n_past_samples))
    estimate_mat = np.zeros((n_past_samples,))
    plot_2d = ax.imshow(sample_mat, vmin=0, vmax=.03, cmap='bone')
    state_est_plot, = plt.plot(estimate_mat, est_color)
    #state_est_plot, = plt.plot(estimate_mat, 'b', lw=3)
    ax.set_ylim(discard_edge_n, N_THETA-discard_edge_n)
    ax.set_xlim(0, n_past_samples)
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('DOA')
    return plot_2d, state_est_plot, sample_mat, estimate_mat

def update_2d_plot(distr, plot_2d, state_est_plot, sample_mat, estimate_mat):
    distr -= np.min(distr)
    distr /= (np.sum(distr) + consts.EPS)
    # UPdate sample_matrix
    sample_mat[:, :-1] = sample_mat[:, 1:]
    sample_mat[:, -1] = distr
    # Update estimate matrix
    maxind = np.argmax(distr)
    estimate_mat[:-1] = estimate_mat[1:]
    estimate_mat[-1] = maxind
    plot_2d.set_array(sample_mat)
    state_est_plot.set_ydata(estimate_mat)

def setup_video_handle(m, n):
    """
    Setup handles for plotting distribution on top of video
    :param m: video height
    :param n: video width
    Returns image plot handle, overlay plot handle
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    implot_h = ax.imshow(np.ones((m, n, 3)))
    # Setup distribution plot handle
    theta_space = np.linspace(0, n, N_THETA)
    plot_h, = ax.plot(theta_space, np.zeros((N_THETA)), 'b', lw=5)
    ax.set_xlim(n, 0)
    ax.set_ylim(m, 0)
    return implot_h, plot_h

def overlay_distribution(image_handle, plot_handle, cvimage, distr):
    image = cvimage[:, :, ::-1]  # Open cv does BGR for some reason
    m, n, _ = image.shape
    if (m, n) != image_handle.get_size():
      sys.stderr.write("ERROR: Given image size is not same as image handle size")
      return
    dist_scale = .5
    dist_offset = .25  # Offset fraction from bottom of frame
    dist = m * (1 - dist_offset) - m * dist_scale * distr
    # Set data
    plot_handle.set_ydata(dist)
    image_handle.set_array(image)
    return image_handle, plot_handle


def localize():
    global switch_beamforming
    global DO_BEAMFORM
    global save_fig
    # Setup search space
    source_plane = OrientedSourcePlane(SOURCE_PLANE_NORMAL, 
                                       SOURCE_PLANE_UP,
                                       SOURCE_PLANE_OFFSET)
    space = SearchSpace(MIC_LOC, CAMERA_LOC, [source_plane], MIC_FORWARD, MIC_ABOVE)
                                       
    # Setup pyaudio instances
    pa = pyaudio.PyAudio()
    helper = AudioHelper(pa)
    #localizer = GridTrackingLocalizer(mic_positions=mic_layout,
    #                                  search_space=space,
    #                                  source_cov=SOURCE_LOCATION_COV,
    #                                  dft_len=FFT_LENGTH,
    #                                  sample_rate=SAMPLE_RATE,
    #                                  n_theta=N_THETA,
    #                                  n_phi=N_PHI)
    localizer = KalmanTrackingLocalizer(mic_positions=mic_layout,
                                      search_space=space,
                                      mic_forward=MIC_FORWARD,
                                      mic_above=MIC_ABOVE,
                                      trans_mat=STATE_TRANSITION_MAT,
                                      state_cov=STATE_TRANSITION_MAT,
                                      emission_mat=EMISSION_MAT,
                                      emission_cov=EMISSION_COV,
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
    if PLOT_POLAR:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_rlim(0, 1)
        plt.show(block=False)
        # Setup space for plotting in new coordinates
        spher_coords = localizer.get_spher_directions()
        theta = spher_coords[1, :]
        pol_plot, = plt.plot(theta, np.zeros(theta.shape))
        post_plot, = plt. plot(theta, np.zeros(theta.shape), 'green')
        ax.set_ylim(0, 1)
        if DO_BEAMFORM:
            pol_beam_plot, = plt.plot(theta, np.ones(theta.shape), 'red')
    if PLOT_CARTES:
        fig = plt.figure()
        ax = plotting.get_halfpage_axis(fig)
        #ax = fig.add_subplot(111)
        plt.show(block=False)
        # Setup space for plotting in new coordinates
        spher_coords = localizer.get_spher_directions()
        theta = spher_coords[1, :]
        theta = np.linspace(0, 1, theta.shape[0])
        gcc_plots = []
        gcc_shaping_vals = [1, 2, 3, 4, 5]
        for i in gcc_shaping_vals:
            plot, = plt.plot(theta, np.zeros(theta.shape))
            gcc_plots.append(plot)
        pol_plot, = plt.plot(theta, np.zeros(theta.shape), 'r--')
        post_plot, = plt. plot(theta, np.zeros(theta.shape), 'b')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, 1)  # Normalized
        #ax.set_xlabel('Angle $\left(\\frac{1}{\pi}\\right)$')
        #ax.set_ylabel('Normalized GCC')
        if DO_BEAMFORM:
            pol_beam_plot, = plt.plot(theta, np.ones(theta.shape), 'red')
    if PLOT_2D:
        #fig_2d = plt.figure(figsize=(10, 6))
        fig = plt.figure()
        w, h = fig.get_size_inches()
        #fig = plt.figure(figsize=(w, 2*h))
        n_past_samples = 200
        ax1 = fig.add_subplot(111)
        #ax2 = fig.add_subplot(212)
        ax2 = ax1
        plot_2d_2, estimate_plot_2, sample_mat_2, estimate_mat_2 = \
                setup_2d_handle(ax2, n_past_samples, 'r', 'SRP-PHAT', 0 * N_THETA)
        plot_2d_1, estimate_plot_1, sample_mat_1, estimate_mat_1 = \
                setup_2d_handle(ax1, n_past_samples, 'b', '', 0 * N_THETA)
        plt.show(block=False)
    if VIDEO_OVERLAY:
        vc = cv2.VideoCapture(0)
        video_handle, video_plot = setup_video_handle(720, 1280)
        plt.show(block=False)
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
                if save_fig:
                    filename = get_fig_name()
                    fig.savefig(filename, facecolor='white', edgecolor='none')
                    print "Figure saved to %s" % filename
                    save_fig = False
                data = in_buf.read_samples(WINDOW_LENGTH)
                # Perform an stft
                stft.performStft(data)
                # Process dfts from windowed segments of input
                dfts = stft.getDFTs()
                rffts = mat.to_all_real_matlab_format(dfts)
                gccs = []
                #for k in gcc_shaping_vals:
                #    d, energy = localizer.get_distribution_real(
                #            rffts[:, :, 0], 'mcc', k) # Use first hop
                #    gccs.append(d)
                d, energy = localizer.get_distribution_real(rffts[:, :, 0], 'beam') # Use first hop
                def w(cpmat):
                    cpmat /= (np.abs(cpmat + consts.EPS))
                    return cpmat
                post = localizer.get_distribution(rffts[:, :, 0], 'beam')
                #post, bla = localizer.get_distribution_real(rffts[:, :, 0], 'mcc')

                #post = localizer.get_distribution(rffts[:, :, 0])
                ind = np.argmax(d)
                u = 1.5 * direcs[:, ind]  # Direction of arrival
                #if energy < 500:
                    #continue

                # Do beam forming
                if DO_BEAMFORM:
                    align_mat = align_mats[:, :, ind]
                    filtered = beamformer.filter_real(rffts, align_mat)
                    mat.set_dfts_real(dfts, filtered, n_channels=2)

                # Take care of plotting
                if count % 1 == 0:
                    if PLOT_POLAR or PLOT_CARTES:
                        dist = d
                        #dist -= np.min(dist)
                        dist = localizer.to_spher_grid(dist)
                        print post.shape
                        post = localizer.to_spher_grid(post) * 50
                        dist /= np.max(dist)
                        if np.max(dist) > 1:
                          dist /= np.max(dist)
                        if np.max(post) > 1:
                          post /= np.max(post)
                        pol_plot.set_ydata(dist[0, :])
                        post_plot.set_ydata(post[0, :])
                        #for i, plot in enumerate(gcc_plots):
                        #    gcc = gccs[i]
                        #    gcc /= (np.max(gcc) + consts.EPS)
                        #    plot.set_ydata(gccs[i])
                        if DO_BEAMFORM:
                            # Get beam plot
                            freq = 2500.  # Hz
                            response = beamformer.get_beam(
                                align_mat, align_mats, rffts, freq
                            )
                            response = localizer.to_spher_grid(response)
                            if np.max(response) > 1:
                                response /= np.max(response)
                            pol_beam_plot.set_ydata(response[-1, :])
                        plt.draw()
                    if PLOT_2D:
                        # Get unconditional distribution
                        dist = localizer.to_spher_grid(d)
                        update_2d_plot(dist, plot_2d_2, estimate_plot_2, 
                                sample_mat_2, estimate_mat_2)
                        dist = localizer.to_spher_grid(post)
                        update_2d_plot(dist, plot_2d_1, estimate_plot_1, 
                                sample_mat_1, estimate_mat_1)
                        plt.draw()
                    if VIDEO_OVERLAY:
                        post /= np.max(post + consts.EPS)
                        dist = d - np.min(d)
                        dist = dist / np.max(dist + consts.EPS)
                        _, cvimage = vc.read()
                        overlay_distribution(video_handle, video_plot, cvimage, dist[::-1])
                        plt.draw()
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


