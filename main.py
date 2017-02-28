#!/usr/bin/python3

import csv
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
import matplotlib.pyplot as plt

CHANNELS = 1
FORMAT = pyaudio.paInt16 # affects size of recorded date
FORMAT = pyaudio.paFloat32
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

def gen(hz, phase, scale):
    print("Generating hz: ", hz)
    rads_per_second = 2.0 * math.pi * hz
    dt = 1/RATE
    last_time = time.time()
    while True:
        now = time_now()
        diff = (now - last_time)/1000.0
        cycles = diff/RATE
        phase = phase + dt * rads_per_second
        value = int(math.cos(phase) * scale)
        #phase = phase + dt * rads_per_second
        if phase >= 2.0 * math.pi:
            phase = phase - 2.0 * math.pi
        last_time = time_now()
        yield value

def generate_signal(hz, length):
    #length = int(length * RATE)
    rads_per_second = float(hz) * math.pi * 2.0
    rads_per_sample = rads_per_second / RATE
    phase = math.pi
    last_time = time.time()
    while True:
        now = time.time()
        diff = (now - last_time)
        phase += rads_per_second * diff
        #phase += factor * length
        arr = numpy.arange(length) # 0,1...
        arr = arr * rads_per_sample
        arr += phase
        chunk = numpy.sin(arr)
        last_time = time.time()
        #plt.plot(arr, chunk)
        #plt.show()
        #chunk *= 32767.0/10.0
        chunk *= 0.5
        yield chunk

def gen2(samples):
    print("Generating #", samples)
    hz = rfftfreq(samples, 1/RATE)[20]
    l = [0 for _ in range(samples)]
    volume = 600000.0
    phase = 0.0
    dt = float(1/RATE)
    rads_per_second = 2.0 * math.pi * float(hz)
    seconds = samples * dt
    rads = rads_per_second * seconds
    last_time = time_now()
    while True:
        now = time_now()
        diff = (now - last_time)/1000.0
        phase += rads_per_second * diff
        l[20] = cmath.rect(volume, phase)
        ret = irfft(l)
        yield ret

# TODO rename
def foo(signal, timestep):
    signal = list(map(lambda x : x / 32767.0, signal))
    signal = numpy.array(signal)
    fft = rfft(signal)
    freqs = zip(fft,
        rfftfreq(len(signal), timestep),
        map(lambda x : abs(x), fft),
        map(lambda x : cmath.phase(x), fft))
    print(max(freqs, key=lambda x : x[2]))
    return
    filtered = filter(lambda x : x[2] > 0.0, freqs)
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
    signal_gen = generate_signal(300, 400)
    signal_gen2 = generate_signal(1024, RATE * 5)
    with pyAudioManager() as p:
        with openStream(p, chunk_size, in_device, out_device) as instream:
            while True:
                new_signal = next(signal_gen)
                new_signal2 = next(signal_gen2)
                s = new_signal2
                d = s.astype(numpy.float32).tostring()
                instream.write(d)
                continue

                data = array.array('f')
                f = 0
                incr = 0.1
                for i in range(0, RATE * 5):
                    #data.append(random.randint(-2000, 2000))
                    f = f + incr
                    #if f > 100 or f < 1:
                    #    incr = -incr
                    #data.append(int(math.sin(f*float(i)/float(chunk_size)) * 1000))
                    data.append(math.sin(f))
                data = data.tobytes()
                instream.write(data)
                return




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
    with pyAudioManager() as p:
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
    chunk_size = audio_reader.chunk_size(1024)
    chunk_time = audio_reader.chunk_time_milli(RATE, 2, chunk_size)
    print("Chunk size: ", chunk_size)
    print("Chunk time: ", chunk_time)
    play_and_record(in_device, out_device, chunk_size, chunk_time)

if __name__ == "__main__":
    main()
