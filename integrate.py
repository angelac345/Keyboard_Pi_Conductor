import pygame
from pygame.time import Clock
import threading
from PythonPiano.key_midi import AudioRecorder
from Gesture_IR.record_video import VideoRecorder
from Gesture_IR.gesture_ir import gesture_ir_main
from IR_Midi.modify_input import modify_volume, convert_midi_to_audio, play_audio_file

from GUI import init_system, GUI 
from metadata import * 
import time 
from metronome import Metronome 


current_frame = None 

class DataFrame: 
    def __init__(self): 
        self.audio_recorder = AudioRecorder(save_path=RAW_MIDI_PATH)
        self.bpm = MIDI_BPM
        self.video_recorder = VideoRecorder(RECORDED_VIDEO_PATH, bpm=self.bpm)

df = DataFrame() 

def start_audio_playback_recording_and_camera(frame, audio_file, video_path):
    df.video_recorder.initialize(df.bpm*2)
    df.video_recorder.spawn_camera()
    time.sleep(2.0)

    playback_thread = threading.Thread(target=play_audio_file, args=(audio_file,))

    frame.idle_components = []
    frame.add_text(
        text="START CONDUCTING TO MUSIC", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )
    frame.event_components = []
    frame.reset() 
    frame.render() 

    playback_thread.start()
    df.video_recorder.start_recording()

    playback_thread.join()
    df.video_recorder.stop_recording() 

    frame.add_text(
        text="MUSIC FINISHED", 
        center_frac_x=0.5, 
        center_frac_y=0.5
    )
    frame.event_components = []
    frame.reset() 
    frame.render() 


def start_production(display_message):
    # Replace with the path to your audio file
    audio_path = 'testrun1.wav'
    video_path = 'testrun1_conduct.mp4'
    midi_path = "mid.midi"
    out_midi_path = 'testrun1_final_midi.mid'
    out_audio_path = 'testrun1_final_audio.wav'
    df.bpm = 90
    beat = '1/4'
    single_note = True


    # print("Hit enter to start recording audio: ") 
    # input()
    # record_audio(20, save_path=audio_path) 

    # print("Hit ENTER to convert the audio to midi file")
    # input()
    # audio_to_midi_conv(audio_path, midi_path, beat=beat, df.bpm=df.bpm)

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
    modify_volume(midi_path, volume_changes, out_midi_path, bpm=df.bpm)

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

def start_recording_audio(frame, next_frame):  
    if frame.get_text()[0] != '': 
        df.bpm = int(frame.get_text()[0])
    metronome.reset_fps(df.bpm/30)
    print('recording with df.bpm: ', df.bpm)

    df.audio_recorder.initialize(bpm=df.bpm)
    switch_frame(next_frame)

def stop_recording_audio(next_frame): 
    print('clicked stop recording audio')
    df.audio_recorder.save_recording()
    switch_frame(next_frame)

def save_recording_continue_cb(next_frame): 
    convert_midi_to_audio(RAW_MIDI_PATH, RAW_AUDIO_PATH)
    switch_frame(next_frame)

def start_conducting_cb(frame, next_frame): 
    frame.add_text(
        text="START CONDUCTING", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )

    frame.reset() 
    frame.render()
    
    start_audio_playback_recording_and_camera(frame, RAW_AUDIO_PATH, RECORDED_VIDEO_PATH)

    frame.idle_components = [] 
    frame.add_text(
        text="RECORDING DONE", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )
    frame.reset() 
    frame.render()

    time.sleep(1)
    frame.idle_components = [] 
    frame.add_button(
        "Start Conducting", 
        0.4, 0.0625, 0.3, 0.4, 
        callback=lambda: start_conducting_cb(start_record_video_frame, combine_frame)
    )

    switch_frame(next_frame)

def generate_result_cb(frame, next_frame): 
    txt, button_rect, cb = frame.event_components[0] 
    frame.event_components[0] = 'Converting...', button_rect, None 
    frame.reset() 
    frame.render() 

    # Convert gesture to IR
    ir_info, ir_data = gesture_ir_main(RECORDED_VIDEO_PATH)

    # Modify MIDI file based on IR and convert to audio
    volume_changes = ir_data
    modify_volume(RAW_MIDI_PATH, volume_changes, MODIFIED_MIDI_PATH, bpm=df.bpm)

    convert_midi_to_audio(MODIFIED_MIDI_PATH, MODIFIED_AUDIO_PATH)

    frame.event_components[0] = txt, button_rect, cb 
    frame.reset() 
    frame.render() 

    switch_frame(next_frame)

def play_result_cb(frame, button_idx): 
    txt, button_rect, cb = frame.event_components[button_idx] 
    frame.event_components[button_idx] = 'Playing Audio...', button_rect, cb 
    frame.reset() 
    frame.render() 

    play_audio_file(MODIFIED_AUDIO_PATH)

    frame.event_components[button_idx] = txt, button_rect, cb 
    frame.reset() 
    frame.render()

def start_over_cb(): 
    start_record_video_frame = GUI(window_width, window_height) 
    start_record_video_frame.add_button(
        "Start Conducting", 
        0.4, 0.0625, 0.3, 0.4, 
        callback=lambda: start_conducting_cb(start_record_video_frame, combine_frame)
    )

    switch_frame(start_frame) 

def re_conduct_cb(): 
    start_record_video_frame = GUI(window_width, window_height) 
    start_record_video_frame.add_button(
        "Start Conducting", 
        0.4, 0.0625, 0.3, 0.4, 
        callback=lambda: start_conducting_cb(start_record_video_frame, combine_frame)
    )

    switch_frame(start_record_video_frame)


init_system() 

window_width = 800
window_height = 600

# Combine done, playback results: 
playback_frame = GUI(window_width, window_height) 
play_idx = playback_frame.add_button(
    "Play Result", 
    0.5, 0.0625, 0.25, 0.4, 
    callback=lambda: play_result_cb(playback_frame, play_idx)
)

playback_frame.add_button(
    "Produce Another Audio", 
    0.5, 0.0625, 0.25, 0.6, 
    callback=lambda: start_over_cb()
)

playback_frame.add_button(
    "Re-Record Conducting Video", 
    0.5, 0.0625, 0.25, 0.8, 
    callback=lambda: re_conduct_cb()
)

# recording stops, click button to combine
combine_frame = GUI(window_width, window_height)
combine_frame.add_button(
    "Combine Video Dynamics with Audio", 
    0.7, 0.0625, 0.15, 0.4, 
    callback=lambda: generate_result_cb(combine_frame, playback_frame)
) 


# frame to start recording video then convert to intermediate representation
start_record_video_frame = GUI(window_width, window_height) 
start_record_video_frame.add_button(
    "Start Conducting", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: start_conducting_cb(start_record_video_frame, combine_frame)
)

# Giving option to re-record audio or proceed: 
re_record_audio_frame = GUI(window_width, window_height) 
go_conducting_button = re_record_audio_frame.add_button(
    "Continue", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: save_recording_continue_cb(start_record_video_frame)
)

re_record_button = re_record_audio_frame.add_button(
    "Re-Record Audio", 
    0.4, 0.0625, 0.3, 0.6, 
    callback=lambda: switch_frame(record_audio_frame)
)

# frame when audio recording is in progress and waiting to stop
audio_recording_frame = GUI(window_width, window_height) 
record_audio_button = audio_recording_frame.add_button(
    "Stop Recording", 
    0.4, 0.0625, 0.3, 0.8, 
    callback=lambda: stop_recording_audio(re_record_audio_frame)
)

audio_recording_frame.add_text(
    text="Hit ENTER to stop and save recording", 
    center_frac_x=0.5, 
    center_frac_y = 0.25
)
audio_recording_frame.add_circle(radius=50, center_frac_x=0.5, center_frac_y=0.5) 
metronome = Metronome(frame=audio_recording_frame, fps=2)

audio_recording_frame.add_per_frame(metronome.switch)
audio_recording_frame.add_event_processor(df.audio_recorder.process_event)


# frame for starting to record audio
record_audio_frame = GUI(window_width, window_height)
record_audio_button = record_audio_frame.add_button(
    "Start Recording", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: start_recording_audio(record_audio_frame, audio_recording_frame) 
)
record_audio_frame.add_textbox(
    0.4, 0.0625, 0.3, 0.7
)

record_audio_frame.add_text(
    text="Click button to start recording audio", 
    center_frac_x=0.5, 
    center_frac_y=0.25
)

# starting screen
start_frame = GUI(window_width, window_height)
start_frame.add_button(
    "Start Production", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: switch_frame(record_audio_frame)
)

clk = Clock()
running = True
current_frame = start_frame
while running: 
    
    for event in pygame.event.get(): 
        exit = current_frame.event_trigger(event) 
        if exit: 
            running = False
        
    current_frame.reset() 
    current_frame.render()
    clk.tick(current_frame.fps) 

pygame.quit()


# if __name__ == '__main__': 
#     main()