__author__ = 'adamjmiller'
from pa_tools.audiobuffer import AudioBuffer
from pa_tools.audiohelper import AudioHelper
from pa_tools.stftmanager import StftManager
import pyaudio
import time
import math
import struct

# Setup data buffer
data_buf = ""
data_played = False
sinbuf = []
for i in range(1024):
    sinbuf.append(math.sin(440*2*math.pi*i/44100))
sinbufstr = struct.pack("1024d",*sinbuf)
record_num = 100 * 1024
num_recorded = 0
in_stream_done = False
out_stream_done = False
tracker = 0
sample_rate = 44100

def in_callback(in_data, frame_count, time_info, status):
    global data_buf
    global num_recorded
    if num_recorded < record_num:
        data_buf = data_buf + in_data
        num_recorded += frame_count
    else:
        return None, pyaudio.paComplete
    return None, pyaudio.paContinue


def out_callback(in_data, frame_count, time_info, status):
    global data_buf
    global tracker
    if tracker < len(data_buf):
        out_data = data_buf[tracker:tracker+frame_count*4]
        tracker += frame_count * 4
        return out_data, pyaudio.paContinue
    else:
        return None, pyaudio.paComplete



pa = pyaudio.PyAudio()
helper = AudioHelper(pa)

# Select device
in_device = helper.get_input_device_from_user()
out_device = helper.get_default_output_device_info()

# Setup stream
in_stream = pa.open(rate=sample_rate,
                    channels = 1,
                    format=pyaudio.paFloat32,
                    input=True,
                    input_device_index=int(in_device['index']),
                    stream_callback=in_callback)
in_stream.start_stream()

print "Recording..."
while in_stream.is_active():
    time.sleep(0.1)
in_stream.close()

out_stream = pa.open(rate=sample_rate,
                     channels=1,
                     format=pyaudio.paFloat32,
                     output=True,
                     output_device_index=int(out_device['index']),
                     stream_callback=out_callback)

out_stream.start_stream()
print "Playing back..."
while out_stream.is_active():
    time.sleep(0.1)

# Cleanup
print "Cleaning up"
out_stream.stop_stream()
out_stream.close()
pa.terminate()

print "Done"




