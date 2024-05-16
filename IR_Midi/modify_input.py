from music21 import converter, stream, tempo, meter

import midi2audio
import pygame
import numpy as np


def convert_midi_to_audio(midi_file, audio_file):
    """
    Convert a MIDI file to an audio file using midi2audio.

    Args:
        midi_file (str): Path to the MIDI file.
        audio_file (str): Path to save the audio file.
    """
    fs = midi2audio.FluidSynth(
        sound_font='/usr/share/sounds/sf2/FluidR3_GM.sf2')
    fs.midi_to_audio(midi_file, audio_file)


def play_audio_file(audio_file):
    """
    Play an audio file using pygame.

    Args:
        audio_file (str): Path to the audio file.
    """
    print("start play audio")
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)
    pygame.mixer.quit()


def modify_volume(midi_file, volume_changes, out_midi_file, bpm=60):
    """
    Modify the volume of a MIDI file based on user input.

    Args:
        midi_file (str): Path to the MIDI file.
        volume_changes (list): List of tuples containing (start time, end time) and volume change.

    Returns:
        music21.stream.Stream: Modified MIDI score.
    """
    # Load the MIDI file
    score = converter.parse(midi_file)

    flat_score = score.flat
    # bpm=score.flat.getElementsByClass(tempo.MetronomeMark)[0].number

    for note in flat_score.notes:
        print(f'note offset: {note.offset}')

    volume = 0
    low, high = 0, 0
    for _, _, volume_change in volume_changes:
        volume += volume_change
        low = min(low, volume)
        high = max(high, volume)

    for i in range(len(volume_changes)):
        t1, t2, volume_change = volume_changes[i]
        volume_changes[i] = (t1, t2, volume_change / (high - low) * 87)

    i = 0
    volume = 0 - low + 40

    note_volume_list = []
    while flat_score.notes[i].offset < volume_changes[i][0]:
        note_volume_list.append(volume)
        i += 1

    buffer = []

    for start_time_seconds, end_time_seconds, volume_change in volume_changes:
        end_offset = bpm * end_time_seconds / 60

        while i < len(flat_score.notes) and flat_score.notes[i].offset < end_offset:
            buffer.append(flat_score.notes[i])
            i += 1

        for note in buffer:
            volume += volume_change / len(buffer)
            note_volume_list.append(volume)

        buffer = []

    while i < len(flat_score.notes):
        note_volume_list.append(volume)
        i += 1

    note_volume_list = np.array(note_volume_list)
    note_volume_list = (note_volume_list - note_volume_list.min()) / \
        (note_volume_list.max() - note_volume_list.min()) * 100 + 20

    print('note_volume_list: ', note_volume_list)
    for i in range(len(note_volume_list)):
        flat_score.notes[i].volume.velocity = note_volume_list[i]

    score.write('midi', out_midi_file)


def test_volume_modification():
    volume_changes = [
        (3.0, 8.0, 130),  # Louder at 7s, lasting for 3s
        (10.0, 380.0, -100)  # Quieter at 15s, lasting for 3s
    ]

    modified_score = modify_volume(
        '/home/pi/Documents/ECE5725_final_proj/output_file.mid', volume_changes, 'output_file.mid')

    # Convert the modified MIDI file to an audio file
    audio_file = 'output_audio.wav'
    convert_midi_to_audio('output_file.mid', audio_file)

    # Play the audio file
    play_audio_file(audio_file)


if __name__ == "__main__":
    test_volume_modification()
    # import sys
    # play_audio_file(sys.argv[1])
