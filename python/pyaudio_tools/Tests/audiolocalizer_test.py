__author__ = 'Adam Miller'

from pa_tools.audiobuffer import AudioBuffer

__author__ = 'adamjmiller'
import unittest
import pa_tools
from pa_tools.audiolocalizer import AudioLocalizer
from pa_tools.directionlocalizer import DirectionLocalizer
from pa_tools.distributionlocalizer import DistributionLocalizer
from pa_tools.stftmanager import StftManager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import scipy.fftpack as fftp
import random
import time
import pa_tools.mattools as mat

class AudioLocalizerTest(unittest.TestCase):

    def setUp(self):
        self.sampling_rate = 44100
        self.dirloc = DirectionLocalizer(mic_layout=None,
                                  sample_rate=44100)
        self.dft_len = 8
        self.stft = StftManager(dft_length=self.dft_len,
                                window_length=self.dft_len,
                                hop_length=self.dft_len,
                                use_window_fcn=False)
        pass

    def testGetPeaks(self):
        g = np.array([[1, 2, 2, 1, 1, 2, 3, 4],
                      [2, 3, 4, 1, 2, 2, 1, 1],
                      [1, 1, 2, 3, 4, 1, 2, 2],
                      [1, 2, 2, 1, 1, 2, 3, 4]])
        G = fftp.ifft(g)
        shift_max = 4
        shift_n = 2 * shift_max + 1
        loc = DirectionLocalizer(mic_layout=None, shift_n=shift_n, shift_max=shift_max)
        peaks = loc.get_peaks(G)
        print peaks
        max_ind = np.argmax(peaks, 1)
        shifts = peaks[0, max_ind]
        self.assertListFloatEqual(np.array([4, 3, -3, 0]), shifts)

    def testGetPeaksSame(self):
        sample_rate = 44100
        loc = DirectionLocalizer(mic_layout=None, sample_rate=sample_rate, shift_n=20, shift_max=2)
        data = np.array([1, -2, 3, 4, 0, 0, 1, 2], dtype=np.float32)
        fft = fftp.fft(data)
        ffts = np.array([fft, fft])
        peaks = loc.get_peaks(ffts)
        inds = np.argmax(peaks, 1)
        delays = peaks[0, inds[1:]]
        delays *= pa_tools.SPEED_OF_SOUND / sample_rate
        self.assertListFloatEqual([0.0], delays)

    def testGetDirectionOrthogonal(self):
        sample_rate = 44100
        mics = np.array([[-.025], [.025]], dtype=np.float32)
        source_loc = np.array([10])
        dist_1 = np.linalg.norm(source_loc - mics[0, :], 2)
        dist_2 = np.linalg.norm(source_loc - mics[1, :], 2)
        loc = DirectionLocalizer(mic_layout=mics, sample_rate=sample_rate, shift_n=20, shift_max=2)
        data = np.array([1, -2, 3, 4, 0, 0, 1, 2], dtype=np.float32)
        fft = fftp.fft(data)
        ffts = np.array([fft, fft])
        direction = loc.get_direction_np(ffts)
        self.assertListFloatEqual([0.0], direction)

    def testGetDirection3Mic(self):
        sample_rate = 16000
        sample_delay = 3
        # Get side_length of mic triangle so that the sample
        # delay will be an integer if source comes from direction
        # perpendicular to some side of the triangle
        side_length = 2 * sample_delay * pa_tools.SPEED_OF_SOUND / (np.sqrt(3) * sample_rate)
        mics = np.array([[0, side_length / np.sqrt(3)],
                         [side_length / 2, -side_length / (2 * np.sqrt(3))],
                         [-side_length / 2, -side_length / (2 * np.sqrt(3))]])

        # Sides are orthogonal to directions (sqrt(3)/2, 1/2), (-sqrt(3)/2, 1/2), (0, 1)
        data_len = 100
        data1 = np.random.rand(1, data_len)
        if sample_delay > 0:
            data2 = np.concatenate((np.random.rand(1, sample_delay),
                                    [data1[0, :-sample_delay]]), axis=1)
        else:
            data2 = data1
        # Get dfts
        fft1 = fftp.fft(data1[0])
        fft2 = fftp.fft(data2[0])
        loc = DirectionLocalizer(mic_layout=mics, sample_rate=sample_rate, shift_max=data_len / 2, shift_n=100)
        ffts = np.array([fft1, fft1, fft2])

        # Get peaks and direction
        peaks = loc.get_peaks(ffts)
        print "Sample delay from mic 1: " + str(peaks[0, (np.argmax(peaks, 1))[1:]])
        direction = loc.get_direction_np(ffts)
        print "Direction to source: " + str(direction)
        direction /= np.linalg.norm(direction, 2)  # Normalize
        direction *= 10  # Scale for plotting
        print mics

        # Plot
        plt.figure()
        plt.plot(mics[:, 0], mics[:, 1], 'bo')
        plt.quiver(0, 0, direction[0], direction[1], scale=20)
        plt.show()
        #self.assertEquals(0, 1)

    def testAngularMethod(self):
        sample_rate = 16000
        sample_delay = 5
        angle = math.pi / 6
        if abs(math.cos(angle)) > 1e-10:
            dist = sample_delay * pa_tools.SPEED_OF_SOUND / (sample_rate * math.cos(angle))
        else:
            dist = 1
            sample_delay = 0
        print "distance: " + str(dist)
        mics = np.array([[0., 0.], [dist, 0.]], dtype=np.float32)
        data_len = 100
        data1 = np.random.rand(1, data_len)
        if sample_delay > 0:
            data2 = np.concatenate((np.random.rand(1, sample_delay),
                                    [data1[0, :-sample_delay]]), axis=1)
        else:
            data2 = data1
        # Get dfts
        fft1 = fftp.fft(data1[0])
        fft2 = fftp.fft(data2[0])
        ffts = np.array([fft1, fft2])
        loc = DirectionLocalizer(mics, sample_rate=sample_rate)
        direction = loc.get_direction_np(ffts)
        print "direction: " + str(direction)

        # Plot
        plt.figure()
        plt.plot(mics[:, 0], mics[:, 1], 'bo')
        plt.quiver(0, 0, direction[0], direction[1], scale=20)
        plt.show()

        #self.assertEquals(0, 1)

    def testifftMatrix(self):
        ffts = np.array([[1, 0, 0, 0],
                        [2, 0, 0, 0],
                        [3, 0, 0, 0],
                        [4, 0, 0, 0]], dtype=np.float32)
        ifft = fftp.ifft(ffts)
        print ifft
        #self.assertEquals(0, 1)



    def testGetDistribution3D(self):
        R = 0.0375
        H = 0.07
        x = np.array([[0, 0, H],
                        [R, 0, 0],
                        [R*math.cos(math.pi/3), R*math.sin(math.pi/3), 0],
                        [-R*math.cos(math.pi/3), R*math.sin(math.pi/3), 0],
                        [-R, 0, 0],
                        [-R*math.cos(math.pi/3), -R*math.sin(math.pi/3), 0],
                        [R*math.cos(math.pi/3), -R*math.sin(math.pi/3), 0]])
        nmics = 7

        # Setup plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.show(block=False)

        # Peform simulation
        N_trials = 10
        for n in range(N_trials):
            # Get random direction
            source = np.array([-100, -100, 0]) + 200 * np.random.rand(3)
            #source =np.array([70, 20, 100])
            # Compute distances and delays
            d = np.sqrt(np.sum((x - source) ** 2, axis=1))
            #d1 = d[0] - d
            delays = d / pa_tools.SPEED_OF_SOUND
            print "delays: " + str(delays)

            # Create audio sample
            Fs = 44100
            T = 1. / Fs
            nsecs = .25
            N = Fs * nsecs
            fund_freq = 50
            low_freq = 1 / (2 * nsecs)
            n = (np.tile(np.arange(N) * T, (nmics, 1)).T - delays).T
            s = np.sin(n * math.pi * low_freq)
            # Add different harmonics to signal
            for k in range(50):
                if k % 3 == 1:
                    s += 5 * np.sin(n * 2 * math.pi * fund_freq * k)
            # Add random noise to each signal
            #s += .35 * np.random.rand(nmics, s.shape[1])

            # Setup localizer
            window_len = 512
            N_THETA = 20
            N_PHI = N_THETA / 2
            loc = DistributionLocalizer(x, sample_rate=Fs, n_theta=N_THETA, dft_len=window_len, n_phi=N_PHI)

            # Get section of signal
            ind = round(random.random() * (N - 512 - 1))
            #ind = 200;
            g = s[:, ind:ind + window_len]
            #print g
            #f = plt.figure()
            #a = f.add_subplot(111)
            #a.plot(np.arange(g.shape[1]), g.T)
            #plt.show()
            G = np.fft.fft(g, n=window_len, axis=1)
            G_real = np.fft.rfft(g, n=window_len, axis=1)

            d_real = loc.get_distribution_real(G_real)
            d = loc.get_distribution_mat(G)
            for i in range(d.shape[0]):
                self.assertListFloatEqual(d[i, :], d_real[i, :])
            print "max: " + str(np.max(d[3, :]))
            print "min: " + str(np.min(d[3, :]))
            maxind = np.argmax(d[3, :])
            u = 1.5 * d[0:3, maxind]
            v = 1.5 * source / np.linalg.norm(source, 2)
            plt.cla()
            ax.scatter(d[0, :], d[1, :], d[2, :], s=30, c=d[3, :])
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]])
            ax.plot([0, u[0]], [0, u[1]], [0, u[2]], c='r')
            #ax.view_init(azim=-90, elev=90)
            plt.draw()
            time.sleep(.5)
        #self.assertEquals(0, 1)

    def assertListFloatEqual(self, list1, list2):
        if not len(list1) == len(list2):
            raise AssertionError("Lists differ in lenght. Cannot be equal")
        for i in range(len(list1)):
            try:
                self.assertLessEqual(abs(list1[i] - list2[i]), 1e-4)
            except AssertionError:
                err_str = "Lists differ on element " + str(i) + ": " + \
                          str(list1[i]) + " vs. " + str(list2[i])
                raise AssertionError(err_str)

    def tearDown(self):
        pass
