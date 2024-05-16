from music21 import converter, stream, tempo, meter, chord, note

import midi2audio
import pygame
from visual_midi import Plotter, Preset
from pretty_midi import PrettyMIDI


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


def seconds_to_offset(seconds, score):
    """
    Convert seconds to offset (quarter note lengths) based on the given tempo.

    Args:
        seconds (float): Time in seconds.
        bpm (int): Beats per minute (tempo).

    Returns:
        float: Offset in quarter note lengths.
    """
    tempo_obj = score.flat.getElementsByClass(tempo.TempoIndication)[0]
    time_signature = score.flat.getElementsByClass(meter.TimeSignature)[0]

    quarter_lengths = tempo_obj.getnaio().secondsToQuarterLength(seconds)

    offset = quarter_lengths / time_signature.ratioString

    return offset


def modify_volume(midi_file, volume_changes):
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
    bpm = score.flat.getElementsByClass(
        tempo.MetronomeMark)[0].number

    for start_time_seconds, end_time_seconds, volume_change in volume_changes:
        start_offset = bpm * start_time_seconds/60
        end_offset = bpm * end_time_seconds/60

        for note in flat_score.notes:
            if start_offset <= note.offset < end_offset:
                print("old vel", note.volume.velocity)
                if volume_change == -100:
                    # If volume_change is -100, set the velocity to 0 (silence)
                    note.volume.velocity = 1
                    print("new val", max(0, min(127, 0)))
                else:
                    new_velocity = int(note.volume.velocity *
                                       (1 + volume_change / 100))
                    print("new val", max(0, min(127, new_velocity)))
                    note.volume.velocity = max(0, min(127, new_velocity))

    return score


def get_note_info(midi_file):
    score = converter.parse(midi_file)
    notes = []
    for element in score.flat.notesAndRests:
        if isinstance(element, stream.Voice):
            element = element.flat.notesAndRests
        if isinstance(element, chord.Chord):
            pitches = '.'.join(n.nameWithOctave for n in element.pitches)
            volume = element.volume.velocity
        elif isinstance(element, note.Note):
            pitches = element.pitch.nameWithOctave
            volume = element.volume.velocity
        else:
            pitches = ''
            volume = 0

        notes.append({
            "pitch": pitches,
            "start": element.offset,
            "end": element.offset + element.duration.quarterLength,
            "volume": volume
        })

    return notes


def draw_note_volumes(screen, notes, y_offset, color):
    for note in notes:
        start_x = int(note["start"] * 50)
        end_x = int(note["end"] * 50)
        volume = int(note["volume"] / 127 * screen.get_height() / 2)
        pygame.draw.rect(screen, color, (start_x, y_offset - volume, end_x - start_x, volume))
        pygame.draw.rect(screen, (0, 0, 0), (start_x, y_offset - volume, end_x - start_x, volume), 1)
        
        font = pygame.font.Font(None, 24)
        pitch_text = font.render(note["pitch"], True, (0, 0, 0))
        volume_text = font.render(str(note["volume"]), True, (0, 0, 0))
        screen.blit(pitch_text, (start_x, y_offset - volume - 20))
        screen.blit(volume_text, (start_x, y_offset - volume - 40))


def draw_modified_notes(screen, original_notes, modified_notes):
    for i in range(len(original_notes)):
        if original_notes[i]["volume"] != modified_notes[i]["volume"]:
            start_x = int(modified_notes[i]["start"] * 50)
            end_x = int(modified_notes[i]["end"] * 50)
            volume = int(modified_notes[i]["volume"] / 127 * screen.get_height() / 2)
            pygame.draw.rect(screen, (255, 0, 0), (start_x, screen.get_height() - volume, end_x - start_x, volume))
            pygame.draw.rect(screen, (0, 0, 0), (start_x, screen.get_height() - volume, end_x - start_x, volume), 1)
            
            font = pygame.font.Font(None, 24)
            pitch_text = font.render(modified_notes[i]["pitch"], True, (0, 0, 0))
            volume_text = font.render(str(modified_notes[i]["volume"]), True, (0, 0, 0))
            screen.blit(pitch_text, (start_x, screen.get_height() - volume - 20))
            screen.blit(volume_text, (start_x, screen.get_height() - volume - 40))



def visualize_notes(original_file, modified_file):
    original_notes = get_note_info(original_file)
    modified_notes = get_note_info(modified_file)

    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("MIDI Note Volume Visualization")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        draw_note_volumes(screen, original_notes,
                          screen_height // 2, (0, 0, 255))
        draw_note_volumes(screen, modified_notes, screen_height, (0, 0, 255))
        draw_modified_notes(screen, original_notes, modified_notes)

        pygame.display.flip()

    pygame.quit()

def visualize_midi(original_file, modified_file):
    original_pm = PrettyMIDI(original_file)
    modified_pm = PrettyMIDI(modified_file)

    preset = Preset(plot_width=800, plot_height=400, row_height=50)
    plotter = Plotter(preset, plot_max_length_bar=16)

    plotter.show(original_pm, "/tmp/original_midi.html")
    plotter.show(modified_pm, "/tmp/modified_midi.html")


def test_volume_modification():
    volume_changes = [
        (3.0, 8.0, 130),  # Louder at 7s, lasting for 3s
        (10.0, 380.0, -100)  # Quieter at 15s, lasting for 3s
    ]

    # modified_score = modify_volume(
    #     'input.mid', volume_changes)

    # modified_score.write('midi', 'output_file.mid')

    # # Convert the modified MIDI file to an audio file
    # audio_file = 'output_audio.wav'
    # convert_midi_to_audio('output_file.mid', audio_file)

    # # Play the audio file
    # play_audio_file(audio_file)

    # Visualize the note volumes
    # visualize_notes('input.mid', 'output_file.mid')

    visualize_midi('input.mid', 'output_file.mid')


if __name__ == "__main__":
    test_volume_modification()
