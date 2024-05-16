import pygame
import threading
from Audio_Midi.audio_to_midi.integrate_main import audio_to_midi_conv
from Audio_Midi.python_midi import record_audio, AudioRecorder
from Gesture_IR.record_video import VideoRecorder
from Gesture_IR.gesture_ir import gesture_ir_main
from IR_Midi.modify_input import modify_volume, convert_midi_to_audio, play_audio_file, play_audio_file

from GUI import init_system, GUI 
from metadata import * 


current_frame = None 
bpm = 90
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

def update_loading(angle):
    screen = pygame.display.get_surface()
    screen.fill((255, 255, 255))
    loading_text = pygame.font.render("Converting video to IR...", True, (0, 0, 0))
    text_rect = loading_text.get_rect(center=(400, 250))
    screen.blit(loading_text, text_rect)
    
    icon_surface = pygame.Surface((50, 50), pygame.SRCALPHA)
    pygame.draw.arc(icon_surface, (0, 0, 0), icon_surface.get_rect(), 0, angle, 5)
    icon_rect = icon_surface.get_rect(center=(400, 300))
    screen.blit(icon_surface, icon_rect)
    
    pygame.display.flip()

def start_conducting_cb(frame, next_frame, bpm): 
    frame.add_text(
        text="START CONDUCTING", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )

    frame.reset() 
    frame.render()
    
    start_audio_playback_recording_and_camera(RECORDED_AUDIO_PATH, RECORDED_VIDEO_PATH, bpm=bpm)

    frame.idle_components = [] 
    frame.add_text(
        text="RECORDING STOPPED", 
        center_frac_x=0.5, 
        center_frac_y=0.25
    )
    frame.reset() 
    frame.render()

    # Convert gesture to IR with rotating loading icon
    angle = 0
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        update_loading(angle)
        angle += 0.1
        if angle >= 360:
            angle -= 360
        
        ir_data = gesture_ir_main(RECORDED_VIDEO_PATH)
        done = True
        
        pygame.time.clock.tick(60)  

    # Modify MIDI file based on IR and convert to audio
    volume_changes = ir_data
    modify_volume(RAW_MIDI_PATH, volume_changes, MODIFIED_MIDI_PATH, bpm=MIDI_BPM)

    convert_midi_to_audio(MODIFIED_MIDI_PATH, MODIFIED_AUDIO_PATH)

    switch_frame(next_frame)


init_system() 
window_width = 800
window_height = 600

# recording stops, click button to combine

combine_frame = GUI(window_width, window_height)
combine_frame.add_button(
    "Combine Video Dynamics with Audio", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: start_conducting_cb(start_record_video_frame, combine_frame)
) 

# frame to start recording video then convert to intermediate representation
start_record_video_frame = GUI(window_width, window_height) 
start_record_video_frame.add_button(
    "Start Conducting", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: start_conducting_cb(start_record_video_frame, combine_frame, bpm)
)

# frame for after audio recording stopped and starting conversion
start_audio_midi_frame = GUI(window_width, window_height) 
start_audio_midi_frame.add_button(
    "Convert Audio to Midi File", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: convert_audio_to_midi_cb(start_audio_midi_frame, start_record_video_frame) 
)

# frame when audio recording is in progress and waiting to stop
audio_recorder = AudioRecorder(save_path=RECORDED_AUDIO_PATH)

audio_recording_frame = GUI(window_width, window_height) 
record_audio_button = audio_recording_frame.add_button(
    "Stop Recording", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: stop_recording_audio(switch_frame=start_audio_midi_frame, recorder=audio_recorder) 
)

audio_recording_frame.add_text(
    text="Click button to stop recording audio", 
    center_frac_x=0.5, 
    center_frac_y = 0.25
)

# frame for starting to record audio
record_audio_frame = GUI(window_width, window_height)
record_audio_button = record_audio_frame.add_button(
    "Start Recording", 
    0.4, 0.0625, 0.3, 0.4, 
    callback=lambda: start_recording_audio(audio_recording_frame, recorder=audio_recorder) 
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

running = True
current_frame = start_frame
while running: 
    
    for event in pygame.event.get(): 
        exit = current_frame.event_trigger(event) 
        if exit: 
            running = False
        
    
    current_frame.reset() 
    current_frame.render()

pygame.quit()


# if __name__ == '__main__': 
#     main()