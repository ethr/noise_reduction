#!/usr/bin/python3


import pyaudio
import wave
import statistics
import struct
import array
import random
import audio_reader
import time
from contextlib import contextmanager
import math
import cmath
import numpy
from numpy.fft import *

CHANNELS = 1
FORMAT = pyaudio.paInt16 # affects size of recorded date
RATE = 44100 # bytes per second
CHUNK = 512 # number of format values in the chuck recorded
RECORD_MSEC = 1
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "file.wav"

@contextmanager
def pyAudioManager():
    audio = pyaudio.PyAudio()
    yield audio
    audio.terminate()

@contextmanager
def openStream(pyAudio, chunk_size, in_device, out_device):
    instream = pyAudio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True, output=True, output_device_index=out_device,
            frames_per_buffer=chunk_size, input_device_index=in_device)
    yield instream
    instream.close()


def time_now():
    return int(round(time.time() * 1000))

def generate_signal(hz, phase, samples, scale):
    print("Generating hz: ", hz)
    print("Generating #", samples)
    ret = []
    rads_per_second = 2.0 * math.pi * hz
    dt = 1/RATE
    for i in range(0, samples):
        value = int(math.cos(phase) * scale)
        ret.append(value)
        phase = phase + dt * rads_per_second
        if phase >= 2 * math.pi:
            phase = phase - 2 * math.pi

    foo(ret, 1/RATE)

    return (ret, phase)

# TODO rename
def foo(signal, timestep):
    signal = list(map(lambda x : x / 32767.0, signal))
    signal = numpy.array(signal)
    fft = rfft(signal)
    freqs = zip(fft,
        rfftfreq(len(signal), timestep),
        map(lambda x : abs(x), fft),
        map(lambda x : cmath.phase(x), fft))
    filtered = filter(lambda x : x[2] > 0.5, freqs)
    filtered = list(filtered)
    for x in filtered:
        print(x)
    #print(list(filtered))
    #print(f)
    #print("fftfreq: ", max(freqs), timestep)
    #print(abs(max(f, lambda x : abs(x[0]))))
    # TODO set a real filter
    #f = list(filter(lambda x : abs(x[0]) > 10000, f))

def play_and_record(in_device, out_device, chunk_size, chunk_time):
    phase = 0
    with pyAudioManager() as p:
        with openStream(p, chunk_size, in_device, out_device) as instream:
            while True:
                start_time = time_now()
                print(phase)
                p = generate_signal(2**9, phase, 2**10, 32767/10)
                #phase = p[1]
                new_signal = p[0]
                new_values = array.array('h')
                new_values.fromlist(new_signal)
                instream.write(new_values.tobytes())
                end_time = time_now()
                dur = (end_time - start_time) / 1000
                rads_per_second = 2.0 * math.pi * 2**9
                phase = phase + dur * rads_per_second
                if (phase >= 2 * math.pi):
                    phase = phase - 2 * math.pi
                continue




                expected_time = start_time + chunk_time
                # start Recording
                #print("recording... ", start_time, " ",expected_time)
                frames = []
                values = array.array('h')
                if True:
                    while len(values) < chunk_size:
                        f = min(512, chunk_size - len(values))
                        try:
                            data = instream.read(num_frames=f)
                        except IOError:
                            pass
                        v = array.array('h', data)
                        values.extend(v)
                        frames.append(data)
                else:
                    data = array.array('h')
                    for i in range(0, chunk_size):
                        #data.append(random.randint(-2000, 2000))
                        f = f + incr
                        if f > 100 or f < 1:
                            incr = -incr
                        data.append(int(sin(f*float(i)/float(chunk_size)) * 1000))
                    data = data.tobytes()

                end_time = time_now()


                print("Values: ", len(values))

                if len(values) == 0 or (end_time == start_time):
                    continue
                #foo(values, (end_time - start_time)/(1000 * 2048))
                foo(values, 1/RATE)

                #for data in frames:
                    #instream.write(data)
def main():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device = p.get_device_info_by_index(i)
        if 'maxOutputChannels' in device and device['maxOutputChannels'] > 0:
            print(i, device['name'])
    out_device = int(input("Select output device #:"))

    for i in range(p.get_device_count()):
        device = p.get_device_info_by_index(i)
        if 'maxInputChannels' in device and device['maxInputChannels'] > 0:
            print(i, device['name'], device['defaultSampleRate'])
    in_device = int(input("Select input device #:"))

    sample_time = audio_reader.sample_time_milli(RATE, 2)
    chunk_size = audio_reader.chunk_size(1024)
    chunk_size = 2048
    chunk_time = audio_reader.chunk_time_milli(RATE, 2, chunk_size)
    print("Chunk size: ", chunk_size)
    print("Chunk time: ", chunk_time)
    print("Sample time: ", sample_time)

#record(WAVE_OUTPUT_FILENAME)
#play_sound(WAVE_OUTPUT_FILENAME)
    play_and_record(in_device, out_device, chunk_size, chunk_time)

if __name__ == "__main__":
    main()
