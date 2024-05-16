import pygame
import threading
from Audio_Midi.audio_to_midi.integrate_main import audio_to_midi_conv
from Audio_Midi.python_midi import record_audio, AudioRecorder
from Gesture_IR.record_video import VideoRecorder
from Gesture_IR.gesture_ir import gesture_ir_main
from IR_Midi.modify_input import modify_volume, convert_midi_to_audio, play_audio_file, play_audio_file

from GUI import init_system, GUI 
from metadata import * 
from utils import *

def start_audio_playback_recording_and_camera(audio_file, video_path, bpm):
    video_recorder = VideoRecorder(video_path=video_path, bpm=bpm) 


    playback_thread = threading.Thread(
        target=play_audio_file, args=(audio_file,))

    
    playback_thread.start()
    print("START CONDUCTING NOW==========")
    video_recorder.start_recording()

    playback_thread.join()
    video_recorder.stop_recording() 
    print('RECORDING STOPPED========')


def start_production(display_message):
    # Replace with the path to your audio file
    audio_path = 'testrun1.wav'
    video_path = 'testrun1_conduct.mp4'
    midi_path = "mid.midi"
    out_midi_path = 'testrun1_final_midi.mid'
    out_audio_path = 'testrun1_final_audio.wav'
    bpm = 90
    beat = '1/4'
    single_note = True


    # print("Hit enter to start recording audio: ") 
    # input()
    # record_audio(20, save_path=audio_path) 

    # print("Hit ENTER to convert the audio to midi file")
    # input()
    # audio_to_midi_conv(audio_path, midi_path, beat=beat, bpm=bpm)

    exit() 
    
    print("Hit ENTER to record conducting video")
    input()
    start_audio_playback_recording_and_camera(
        audio_path, video_path
    )

    display_message("Start production")

    print('Hit ENTER to convert gesture video to IR')
    input()
    # Convert gesture to IR
    ir_data = gesture_ir_main(video_path)

    # Modify MIDI file based on IR and convert to audio
    print('Hit ENTER to convert video IR to dynamics: ')
    input()
    volume_changes = ir_data
    modify_volume(midi_path, volume_changes, out_midi_path,bpm=bpm)

    print('Hit ENTER to convert modified midi file to audio file')
    input()
    convert_midi_to_audio(out_midi_path, out_audio_path)

    display_message("Production finished, let's see the result")


def show_result():
    # Visualize MIDI files and play back the modified audio
    original_midi = "converted_midi.mid"
    modified_midi = "output_file.mid"
    # visualize_notes(original_midi, modified_midi)

    play_audio_file("output_audio.wav")


def display_message(message):
    global current_message
    current_message = message

def switch_frame(switch_frame): 
    print('Switching Frame')
    global current_frame 
    current_frame = switch_frame

def start_recording_audio(switch_frame, recorder): 
    recorder.start_recording()
    global current_frame 
    current_frame = switch_frame

def stop_recording_audio(recorder, switch_frame=None): 
    recorder.stop_recording() 
    global current_frame 
    if switch_frame is not None: 
        current_frame = switch_frame


def convert_audio_to_midi_cb(frame, next_frame): 
    frame.add_text(
        text="Converting...", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )

    frame.reset() 
    frame.render()

    audio_to_midi_conv(RECORDED_AUDIO_PATH, RAW_MIDI_PATH, beat=MIDI_BEAT, bpm=MIDI_BPM)

    frame.idle_components = [] 
    frame.add_text(
        text="DONE", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )
    frame.reset() 
    frame.render()

    switch_frame(next_frame)

def start_conducting_cb(frame, next_frame): 
    frame.add_text(
        text="START CONDUCTING", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )

    frame.reset() 
    frame.render()
    
    start_audio_playback_recording_and_camera(RECORDED_AUDIO_PATH, RECORDED_VIDEO_PATH)

    frame.idle_components = [] 
    frame.add_text(
        text="RECORDING STOPPED", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )
    frame.reset() 
    frame.render()

    # Convert gesture to IR
    ir_data = gesture_ir_main(RECORDED_VIDEO_PATH)

    # Modify MIDI file based on IR and convert to audio
    volume_changes = ir_data
    modify_volume(RAW_MIDI_PATH, volume_changes, MODIFIED_MIDI_PATH, bpm=MIDI_BPM)

    convert_midi_to_audio(MODIFIED_MIDI_PATH, MODIFIED_AUDIO_PATH)

    switch_frame(next_frame)
