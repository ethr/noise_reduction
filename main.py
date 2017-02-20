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
from math import sin

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

def play_and_record(in_device, out_device, chunk_size, chunk_time):
    with pyAudioManager() as p:

        print(RATE)
        print(1000 * chunk_size)
        print(1000 * chunk_size)
        print((RATE / (1000 * chunk_size)) * 50 * RECORD_MSEC)

        with openStream(p, chunk_size, in_device, out_device) as instream:
            f = 1
            incr = 0.1
            while True:
                start_time = time_now()
                expected_time = start_time + chunk_time
                # start Recording
                #print("recording... ", start_time, " ",expected_time)
                frames = []
                try:
                    if True:
                        #for i in range(0, int((RATE / (1000 * chunk_size)) * 50 * RECORD_MSEC)):
                        for i in range(0, 4):
                            data = instream.read(num_frames=512)
                            frames.append(data)
                    else:
                        data = array.array('i')
                        for i in range(0, chunk_size):
                            #data.append(random.randint(-2000, 2000))
                            f = f + incr
                            if f > 100 or f < 1:
                                incr = -incr
                            data.append(int(sin(f*float(i)/float(chunk_size)) * 1000))
                        data = data.tobytes()
                except IOError:
                    print("error", time_now())

                end_time = time_now()
                #print ("stopped recording ", end_time - start_time)

                #print("finished recording")
                #print(len(data))

                #print("playing")
                #open output stream

                #read data

                #play stream
                #print("mean: {} max: {}".format(mean, m))
                values = array.array('h')
                for data in frames:
                    v = array.array('h', data)
                    values.extend(v)
                print("Max of values: ", max(values))
                print("Max of values: ", 100.0 * float(max(values))/65536)
                """
                print(mean)
                print(m)
                for i in range(0, len(values)):
                    values[i] = m - values[i]
                print(values)
                data = values.tobytes()
                #for i in range(0, len(chunk)):
                    #chunk[i] = -chunk[i]
                """

                for data in frames:
                    instream.write(data)
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
    #chunk_size = audio_reader.chunk_size(128)
    chunk_size = 256
    chunk_time = audio_reader.chunk_time_milli(RATE, 2, chunk_size)
    print("Chunk size: ", chunk_size)
    print("Chunk time: ", chunk_time)
    print("Sample time: ", sample_time)

#record(WAVE_OUTPUT_FILENAME)
#play_sound(WAVE_OUTPUT_FILENAME)
    play_and_record(in_device, out_device, chunk_size, chunk_time)

if __name__ == "__main__":
    main()



