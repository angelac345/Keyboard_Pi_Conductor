# from midiutil.MidiFile import
import pyaudio
import wave

from Audio_Midi.audio_to_midi.converter import Converter
from threading import Thread 

class AudioRecorder: 
    def __init__(self, save_path): 
        self.recording = False 
    
        self.audio = pyaudio.PyAudio()


        self.usb_idx = 0

        # Outputs index of every audio-capable device on the Pi
        for ii in range(self.audio.get_device_count()):
            if "USB PnP Sound" in self.audio.get_device_info_by_index(ii).get('name'): 
                self.usb_idx = ii
            print(f"idx: {ii}, name: {self.audio.get_device_info_by_index(ii).get('name')}")

        # Set index of usb microphone

        # Set parameters for audio recording
        self.form_1 = pyaudio.paInt16  # 16-bit resolution
        self.chans = 1  # 1 channel
        self.samp_rate = 44100  # 44.1kHz sampling rate
        self.chunk = 4096  # 2^12 samples for buffer
        self.save_path = save_path  # name of .wav file
        self.frames = []


    def start_recording(self): 
        self.stream = self.audio.open(
            format=self.form_1, 
            rate=self.samp_rate, 
            channels=self.chans,          
            input_device_index=self.usb_idx, 
            input=True,
            frames_per_buffer=self.chunk
        )
        print("recording")

        self.recording = True 
        self.record_thread = Thread(target=self.record)
        self.record_thread.start()

    def record(self): 
        while self.recording: 
            data = self.stream.read(self.chunk)
            self.frames.append(data)    
    
    def stop_recording(self): 
        self.recording = False 

        self.record_thread.join() 
        print('Recording Stopped, closing streams')

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        print("finished recording")

        # save the audio frames as .wav file
        wavefile = wave.open(self.save_path, 'wb')
        wavefile.setnchannels(self.chans)
        wavefile.setsampwidth(self.audio.get_sample_size(self.form_1))
        wavefile.setframerate(self.samp_rate)
        wavefile.writeframes(b''.join(self.frames))
        wavefile.close()


def start_recording(audio, form_1, chans, samp_rate, chunk, usb_idx):
    # create pyaudio stream
    stream = audio.open(format=form_1, rate=samp_rate, channels=chans,
                        input_device_index=usb_idx, input=True,
                        frames_per_buffer=chunk)
    print("recording")
    return stream


def record_loop(stream, samp_rate, chunk, record_secs):
    # loop through stream and append audio chunks to frame array
    frames = []
    for ii in range(int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)
    return frames


def stop_recording(audio, stream, filename, chans, samp_rate, frames, form_1):
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("finished recording")

    # save the audio frames as .wav file
    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()


def audio_to_midi_conv(input_filename, output_filename, time_window=5.0, activ_level=0.0, note_count=0, bpm=60):
    process = Converter(
        infile=input_filename,
        outfile=output_filename,
        time_window=time_window,
        activation_level=activ_level,
        condense=None,
        condense_max=False,
        max_note_length=0,
        note_count=note_count,
        bpm=bpm,
    )
    process.convert()


def record_audio(audio_length, save_path):
    audio = pyaudio.PyAudio()

    # Outputs index of every audio-capable device on the Pi
    for ii in range(audio.get_device_count()):
        print(f"idx: {ii}, name: {audio.get_device_info_by_index(ii).get('name')}")

    # Set index of usb microphone
    usb_idx = 1

    # Set parameters for audio recording
    form_1 = pyaudio.paInt16  # 16-bit resolution
    chans = 1  # 1 channel
    samp_rate = 44100  # 44.1kHz sampling rate
    chunk = 4096  # 2^12 samples for buffer
    wav_output_filename = save_path  # name of .wav file
    frames = []

    stream = start_recording(audio, form_1, chans, samp_rate, chunk, usb_idx)
    frames = record_loop(stream, samp_rate, chunk, audio_length)
    stop_recording(audio, stream, wav_output_filename,
                   chans, samp_rate, frames, form_1)


if __name__ == "__main__":
    # record_audio(10, 'test_recorder.wav')
    recorder = AudioRecorder(save_path='test_recorder.wav') 

    print('Hit Enter to Start Recording')
    input() 
    recorder.start_recording()

    print('Hit Enter to Stop Recording')
    input() 
    recorder.stop_recording() 
    