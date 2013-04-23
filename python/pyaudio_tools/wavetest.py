__author__ = 'Adam Miller'
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
from wavehelper import WaveHelper

N_CHANNELS = 1
SECONDS = 1
SAMPLE_WIDTH = 2  # Bytes
SAMPLE_RATE = 44100
MAX_VAL = 2 ** (SAMPLE_WIDTH * 8 - 1)
N_FRAMES = SAMPLE_RATE * SECONDS
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = (N_CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, N_FRAMES, 'NONE', 'not compressed')
fmts = (None, "B", "h", None, "l")

def make_noise():
    outfile = wave.open('output.wav', 'w')
    outfile.setparams(params)
    fmt = fmts[SAMPLE_WIDTH]

    noise = np.random.randint(-MAX_VAL, MAX_VAL, N_FRAMES)
    for i in range(len(noise)):
        out_frame = struct.pack(fmt, noise[i])
        outfile.writeframes(out_frame)
    outfile.close()

    # Read file back in
    infile = wave.open('output.wav', 'r')
    inframes = infile.readframes(N_FRAMES)
    nfmt = str(N_FRAMES) + 'h'
    print nfmt
    data = struct.unpack(nfmt, inframes)
    plt.plot(data)
    plt.show()

def read_wav():
    # Read in high.wav file
    high = wave.open('high.wav', 'r')
    nframes = high.getnframes()
    inframes = high.readframes(nframes)
    fmt = fmts[high.getsampwidth()]
    nfmt = str(nframes) + fmt
    print nfmt
    data = struct.unpack(nfmt, inframes)
    plt.plot(data)
    plt.show()


def make_sin():
    # Create signal
    frequency = 440.  # Hz
    n_secs = 1.
    n_frames = SAMPLE_RATE * n_secs
    n = np.arange(n_frames)
    sin = np.sin(2 * np.pi * frequency * n / SAMPLE_RATE)
    sin = np.asarray(sin * (MAX_VAL - 1), dtype=np.int16)  # Convert to shorts

    # Create wav file
    n_channels = 1
    frame_width = 2  # Bytes
    comptype = 'NONE'
    compname = 'not compressed'
    parms = (n_channels, frame_width, SAMPLE_RATE, n_frames, comptype, compname)
    sinwav = wave.open('sin.wav', 'w')
    sinwav.setparams(parms)

    # Convert to bytes and write to file
    data = struct.pack('%dh' % n_frames, *sin)
    sinwav.writeframes(data)
    sinwav.close()


def main():
    make_sin()



if __name__ == '__main__':
    main()
