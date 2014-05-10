__author__ = 'adamjmiller'
import wave
import struct
import threading
import math
import cv2

import pyaudio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pa_tools.constants as consts
import mattools.mattools as mat
from pa_tools.audiohelper import AudioHelper
from pa_tools.audiobuffer import AudioBuffer
from pa_tools.commandlistener import CommandListener
from pa_tools.stftmanager import StftManager
from pa_tools.vonmisestrackinglocalizer import VonMisesTrackingLocalizer
from pa_tools.beamformer import BeamFormer
from searchspace import SearchSpace
from searchspace import OrientedSourcePlane
from plottools.particlefilterplot import ParticleFilterPlot
from plottools.plotmanager import PlotManager
import plottools.plottools as ptools


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
THETA_SPACE = np.linspace(0, np.pi, N_THETA)
N_PHI = 1
PLOT_POLAR = False
PLOT_CARTES = False
PLOT_2D = False
EXTERNAL_PLOT = False
PLAY_AUDIO = False
DO_BEAMFORM = False
RECORD_AUDIO = False
VIDEO_OVERLAY = False
SAVE_FRAMES = False
PLOT_PARTICLES = True
OUTFILE_NAME = 'nonbeamformed.wav'
TIMEOUT = 1
# Source planes and search space
SOURCE_PLANE_NORMAL = np.array([0, -1, 0])
SOURCE_PLANE_UP = np.array([0, 0 , 1])
SOURCE_PLANE_OFFSET = np.array([0, 1, 0])
SOURCE_LOCATION_COV = np.array([[1, 0], [0, .01]])
MIC_LOC = np.array([0, 0, 0])
CAMERA_LOC = np.array([0, 0, 0])
TIME_STEP = .1
MIC_FORWARD = np.array([0, 1, 0])
MIC_ABOVE = np.array([0, 0, 1])
STATE_KAPPA = 100  
OUTLIER_PROB = .9
OBS_KAPPA = 10
N_PARTICLES = 50

# Setup printing
np.set_printoptions(precision=4, suppress=True)
#ptools.setup_halfpage_figsize()

# Setup mics
mic_layout = np.array([[.03, 0], [-.01, 0], [.01, 0], [-.03, 0]])
# Track whether we have quit or not

# Global variable used to end pa buffer read/write calls
done = False

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
        return '\x00' * frame_count * SAMPLE_SIZE * \
                NUM_CHANNELS_OUT, pyaudio.paContinue


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
    data = np.asarray(data*.5*SHORT_MAX, dtype=np.int16)
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

def setup_2d_particle_handle(ax, n_past_samples, title='', discard_edge_n = 0):
    # Get rid of grid lines
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Setup structures for maintaining all data to display in the image
    lhood_mat = np.zeros((N_THETA, n_past_samples))
    particles = np.zeros((N_PARTICLES, n_past_samples))
    weights = np.zeros((N_PARTICLES, n_past_samples))
    # Setup image
    im_2d = ax.imshow(lhood_mat, vmin=0, vmax=.03, cmap='bone', origin='lower',
                      extent=[0, n_past_samples-1, 0, np.pi], aspect='auto')
    # Setup particle plots
    #part_plots = ax.plot(np.arange(0, n_past_samples), particles.T, 'r.')
    scatter_space = np.kron(np.ones((N_PARTICLES,)), np.arange(n_past_samples)) 
    part_scat = plt.scatter(scatter_space, 
        np.reshape(particles, N_PARTICLES * n_past_samples), edgecolors='none', facecolors='r')
    ax.set_ylim(0, np.pi)
    ax.set_xlim(0, n_past_samples-1)
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('DOA')
    return im_2d, scatter_space, part_scat, lhood_mat, particles, weights

def update_2d_plot(ax, im_2d, scatter_space, part_scat, lhood_mat, particles, weights, 
                      distr, new_parts, new_weights):
    # Update likelihood matrix
    lhood_mat[:, :-1] = lhood_mat[:, 1:]
    dist = distr - np.min(distr)
    dist /= (np.sum(dist) + consts.EPS)
    lhood_mat[:, -1] = dist
    # Translate particles into theta space
    particles[:, :-1] = particles[:, 1:]
    particles[:, -1] = np.arctan2(new_parts[:, 1], new_parts[:, 0])
    # Update weights
    weights[:, :-1] = weights[:, 1:]
    weights[:, -1] = new_weights
    # Update plot of localization likelihoods
    im_2d.set_array(lhood_mat)
    part_scat.set_offsets(
            np.array([scatter_space, np.reshape(particles, particles.size)]).T)

    #part_scat._sizes = np.reshape(weights, weights.size) * 200

def setup_video_handle(ax, m, n):
    """
    Setup handles for plotting distribution on top of video
    :param m: video height
    :param n: video width
    Returns image plot handle, overlay plot handle
    """
    implot_h = ax.imshow(np.ones((m, n, 3)))
    # Setup distribution plot handle
    particle_plots = []
    offset = m - m * .2
    for i in range(N_PARTICLES):
      particle_plots.append(ax.plot([0], [offset], 'o', mfc='none', mec='r')[0])
    estimate_plot, = ax.plot([0], [offset], 'w+', ms=40)
    ax.set_xlim(n, 0)
    ax.set_ylim(m, 0)
    return implot_h, particle_plots, estimate_plot

def overlay_particles(image_handle, particle_plots, estimate_plot, 
                      cvimage, particles, weights, estimate):
    image = cvimage[:, :, ::-1]  # Open cv does BGR for some reason
    m, n, _ = image.shape
    if (m, n) != image_handle.get_size():
      sys.stderr.write("ERROR: Given image size is not same as image handle size")
      return
    #thetas = np.arctan2(particles[:, 1], particles[:, 0]))
    # Set data -- project onto a plane a given distance away
    distance = 1  # One meter
    xs = distance * particles[:, 0] / particles[:, 1]
    xs = n/2 * (xs + 1)  # Map to pixel n/2
    # Do same for estimate
    estim = n/2 * (distance * estimate[0] / estimate[1] + 1)
    for i, particle_plot in enumerate(particle_plots):
      particle_plot.set_xdata(xs[i])
      particle_plot.set_markersize((weights[i]**3) * 10000000 )
    estimate_plot.set_xdata(estim)
    image_handle.set_array(image)
    return image_handle, particle_plots, estimate_plot

def setup_particle_plot(ax, particle_color, estim_color, offset):
    """
    Setup the particle plot handles
    :param ax: axis handle to plot on
    :param particle_color: matplotlib color for particle colors
    :param estim_color: matplotlib color for estimate color
    :param offset: offset from bottom of screen to level at which particles
                   will be plotted. 0 is bottom of frame, 1 is top
    """
    particle_plots = []
    for i in range(N_PARTICLES):
      particle_plots.append(ax.plot([0], [offset], 'o', mfc='none', mec=particle_color)[0])
    estimate_plot, = ax.plot([0], [offset], c=estim_color, marker='.', ms=20)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1)
    return particle_plots, estimate_plot

def plot_particles(particle_plots, estim_handle, particles, weights, estimate):
    thetas = np.arctan2(particles[:, 1], particles[:, 0])
    for i, particle_plot in enumerate(particle_plots):
      particle_plot.set_xdata(thetas[i])
      particle_plot.set_markersize(weights[i] * 2000)
    estim_handle.set_xdata(np.arctan2(estimate[1], estimate[0]))
    return particle_plots, estim_handle

def localize():
    # Global variables that may be set in this function
    global DO_BEAMFORM
    global done
    # Setup search space
    source_plane = OrientedSourcePlane(SOURCE_PLANE_NORMAL, 
                                       SOURCE_PLANE_UP,
                                       SOURCE_PLANE_OFFSET)
    space = SearchSpace(MIC_LOC, CAMERA_LOC, [source_plane], MIC_FORWARD, MIC_ABOVE)
                                       
    # Setup pyaudio instances
    pa = pyaudio.PyAudio()
    helper = AudioHelper(pa)
    listener = CommandListener()
    plot_manager = PlotManager('vmpf_2d_obskap_25_')
    localizer = VonMisesTrackingLocalizer(mic_positions=mic_layout,
                                      search_space=space,
                                      n_particles=N_PARTICLES,
                                      state_kappa=STATE_KAPPA,
                                      observation_kappa=OBS_KAPPA,
                                      outlier_prob=.5,
                                      dft_len=FFT_LENGTH,
                                      sample_rate=SAMPLE_RATE,
                                      n_theta=N_THETA,
                                      n_phi=N_PHI)
    localizer2 = VonMisesTrackingLocalizer(mic_positions=mic_layout,
                                      search_space=space,
                                      n_particles=N_PARTICLES,
                                      state_kappa=STATE_KAPPA,
                                      observation_kappa=OBS_KAPPA,
                                      outlier_prob=.5,
                                      dft_len=FFT_LENGTH,
                                      sample_rate=SAMPLE_RATE,
                                      n_theta=N_THETA,
                                      n_phi=N_PHI)
    localizer3 = VonMisesTrackingLocalizer(mic_positions=mic_layout,
                                      search_space=space,
                                      n_particles=N_PARTICLES,
                                      state_kappa=STATE_KAPPA,
                                      observation_kappa=OBS_KAPPA,
                                      outlier_prob=.8,
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
    listener.start_polling()

    # Setup directions and alignment matrices
    direcs = localizer.get_directions()
    align_mats = localizer.get_pos_align_mat()

    # Plotting setup
    if PLOT_PARTICLES:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        particle_plots, estimate_plot = setup_particle_plot(ax, 'b', 'r', .2)
        particle_plots2, estimate_plot2 = setup_particle_plot(ax, 'k', 'r', .5)
        particle_plots3, estimate_plot3 = setup_particle_plot(ax, 'g', 'r', .8)
        plt.show(block=False)
    if PLOT_POLAR:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_rlim(0, 1)
        plt.show(block=False)
        # Setup space for plotting in new coordinates
        spher_coords = localizer.get_spher_directions()
        theta = spher_coords[1, :]
        pol_plot, = plt.plot(theta, np.ones(theta.shape))
        post_plot, = plt.plot(theta, np.ones(theta.shape), 'green')
        ax.set_ylim(0, 1)
        if DO_BEAMFORM:
            pol_beam_plot, = plt.plot(theta, np.ones(theta.shape), 'red')
    if PLOT_CARTES:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 1)
        plt.show(block=False)
        # Setup space for plotting in new coordinates
        spher_coords = localizer.get_spher_directions()
        theta = spher_coords[1, :]
        pol_plot, = plt.plot(theta, np.ones(theta.shape))
        post_plot, = plt.plot(theta, np.ones(theta.shape), 'green')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, np.pi)
        if DO_BEAMFORM:
            pol_beam_plot, = plt.plot(theta, np.ones(theta.shape), 'red')
    if PLOT_2D:
        n_past_samples = 100
        estimate_colors = ['b', 'r', 'g', 'k'] # Noisy, estimate, class0, class1
        particle_plot = ParticleFilterPlot(N_PARTICLES, 
            n_space=N_THETA, n_past_samples=n_past_samples, 
            n_estimates=3, particle_color='r', distr_cmap='bone',
            estimate_colors=estimate_colors)
    if VIDEO_OVERLAY:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        vc = cv2.VideoCapture(0)
        video_handle, vid_part_plots, vid_estim_plot = setup_video_handle(ax, 720, 1280)
        plt.show(block=False)
    if EXTERNAL_PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.show(block=False)

    count = 0
    try:
        while in_stream.is_active() or out_stream.is_active():
            done = listener.quit()
            data_available = in_buf.wait_for_read(WINDOW_LENGTH, TIMEOUT)
            if data_available:
                if listener.switch_beamforming():
                    DO_BEAMFORM = not DO_BEAMFORM
                    # Get data from the circular buffer
                data = in_buf.read_samples(WINDOW_LENGTH)
                # Perform an stft
                stft.performStft(data)
                # Process dfts from windowed segments of input
                dfts = stft.getDFTs()
                rffts = mat.to_all_real_matlab_format(dfts)
                d, energy = localizer.get_distribution_real(rffts[:, :, 0], 'gcc') # Use first hop
                post = localizer.get_distribution(rffts[:, :, 0])
                joint_w = localizer.get_joint_weights()
                post2 = localizer2.get_distribution(rffts[:, :, 0])
                post3 = localizer3.get_distribution(rffts[:, :, 0])
                w = np.asarray(post.weights)
                ps = np.asarray(post.particles)
                estimate = w.dot(ps)
                w2 = np.asarray(post2.weights)
                ps2 = np.asarray(post2.particles)
                estimate2 = w2.dot(ps2)
                w3 = np.asarray(post3.weights)
                ps3 = np.asarray(post3.particles)
                estimate3 = w3.dot(ps3)
                #if energy < 1000:
                #    continue

                # Do beam forming
                if DO_BEAMFORM:
                    align_mat = align_mats[:, :, ind]
                    filtered = beamformer.filter_real(rffts, align_mat)
                    mat.set_dfts_real(dfts, filtered, n_channels=2)

                # Take care of plotting
                if count % 1 == 0:
                    if PLOT_PARTICLES:
                        plot_particles(particle_plots, estimate_plot, ps, w, estimate)
                        plot_particles(particle_plots2, estimate_plot2, ps2, w2, estimate2)
                        plot_particles(particle_plots3, estimate_plot3, ps3, w3, estimate3)
                        plt.draw()
                        
                    if PLOT_POLAR or PLOT_CARTES:
                        dist = d
                        #dist -= np.min(dist)
                        dist = localizer.to_spher_grid(dist)
                        post = localizer.to_spher_grid(post) * 50
                        #dist /= np.max(dist)
                        if np.max(dist) > 1:
                          dist /= np.max(dist)
                        if np.max(post) > 1:
                          post /= np.max(post)
                        pol_plot.set_ydata(dist[0, :])
                        post_plot.set_ydata(post[0, :])
                        if DO_BEAMFORM:
                            # Get beam plot
                            freq = 1900.  # Hz
                            response = beamformer.get_beam(align_mat, align_mats, rffts, freq)
                            response = localizer.to_spher_grid(response)
                            if np.max(response) > 1:
                                response /= np.max(response)
                            pol_beam_plot.set_ydata(response[-1, :])
                        plt.draw()
                    if PLOT_2D:
                        dist = localizer.to_spher_grid(d)
                        theta_parts = np.arctan2(ps[:, 1], ps[:, 0])
                        noisy = THETA_SPACE[np.argmax(dist)]
                        estimate = w.dot(theta_parts)
                        spike_estimate = joint_w[0, :].dot(theta_parts) /(np.sum(joint_w[0, :]) + consts.EPS)
                        slab_estimate = joint_w[1, :].dot(theta_parts) / (np.sum(joint_w[1, :]) + consts.EPS)
                        particle_plot.update(dist, theta_parts, w, 
                            [noisy, estimate, spike_estimate])#, slab_estimate])
                        if listener.savefig():
                            plot_manager.savefig(particle_plot.get_figure())
                    if VIDEO_OVERLAY:
                        _, cvimage = vc.read()
                        overlay_particles(video_handle, vid_part_plots, vid_estim_plot, \
                                              cvimage, ps, w, estimate)
                        plt.draw()
                    if SAVE_FRAMES:
                        fig.canvas.print_rgba('out/out' + str(count) + '.mat')
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
        listener.set_quit(True)


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



