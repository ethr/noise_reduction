import pyaudio
import wave
import statistics
import struct
import array
import random
import audio_reader
import time
from math import sin

CHANNELS = 1
FORMAT = pyaudio.paInt16 # affects size of recorded date
RATE = 44100 # bytes per second
CHUNK = 512 # number of format values in the chuck recorded
RECORD_MSEC = 1
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "file.wav"

def time_now():
    return int(round(time.time() * 1000))

def record(path_to_file):

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            frames_per_buffer=CHUNK, input_device_index=2)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")


    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(path_to_file, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()



def play_sound(path_to_file):
    print("Playing sound")

    #open a wav format music
    f = wave.open(path_to_file)
    #instantiate PyAudio
    p = pyaudio.PyAudio()
    #open stream
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
            channels = f.getnchannels(),
            rate = f.getframerate(),
            output_device_index=2,
            output = True)
    #read data
    data = f.readframes(CHUNK)

    #play stream
    while data != '':
        stream.write(data)
        data = f.readframes(CHUNK)

    #stop stream
    stream.stop_stream()
    stream.close()

    #close PyAudio
    p.terminate()


def play_and_record(in_device, out_device, chunk_size, chunk_time):
    #instantiate PyAudio
    p = pyaudio.PyAudio()

    try:

        f = 1
        incr = 0.1
        instream = p.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True, output=True, output_device_index=out_device,
                frames_per_buffer=chunk_size, input_device_index=in_device)
        while True:
            start_time = time_now()
            expected_time = start_time + chunk_time
            # start Recording
            print("recording... ", start_time, " ",expected_time)
            data = []
            #for i in range(0, int((RATE / (1000 * CHUNK)) * 50 * RECORD_MSEC)):
            try:
                if True:
                    data = instream.read(num_frames=chunk_size)
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
                end_time = time_now()
                print("error", end_time)
                if end_time < expected_time:
                   print("sleeping")
                   time.sleep((expected_time - end_time)/1000) 
                continue
            #print("finished recording")
            #print(len(data))

            #print("playing")
            #open output stream

            #read data

            #play stream
            #print("mean: {} max: {}".format(mean, m))
            """
            values = array.array('i', data)
            mean = statistics.mean(data)
            m = max(data)
            print(mean)
            print(m)
            for i in range(0, len(values)):
                values[i] = m - values[i]
            print(values)
            data = values.tobytes()
            #for i in range(0, len(chunk)):
                #chunk[i] = -chunk[i]
            """
            instream.write(data)

            end_time = time_now()
            if end_time < expected_time:
               print("sleeping")
               #time.sleep((expected_time - end_time)/1000) 
            #stop stream
    finally:
        if instream is not None:
            instream.stop_stream()
            instream.close()
        #close PyAudio
        p.terminate()

def main():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device = p.get_device_info_by_index(i)
        for key in device:
           print(key)
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



