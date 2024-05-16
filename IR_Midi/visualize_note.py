from music21 import converter, stream, tempo, meter, chord, note

import midi2audio
import pygame
from IR_Midi.modify_input import modify_volume
from visual_midi import Plotter, Preset
from pretty_midi import PrettyMIDI


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
        pygame.draw.rect(screen, color, (start_x, y_offset -
                         volume, end_x - start_x, volume))
        pygame.draw.rect(screen, (0, 0, 0), (start_x,
                         y_offset - volume, end_x - start_x, volume), 1)

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
            volume = int(modified_notes[i]["volume"] /
                         127 * screen.get_height() / 2)
            pygame.draw.rect(screen, (255, 0, 0), (start_x,
                             screen.get_height() - volume, end_x - start_x, volume))
            pygame.draw.rect(screen, (0, 0, 0), (start_x, screen.get_height(
            ) - volume, end_x - start_x, volume), 1)

            font = pygame.font.Font(None, 24)
            pitch_text = font.render(
                modified_notes[i]["pitch"], True, (0, 0, 0))
            volume_text = font.render(
                str(modified_notes[i]["volume"]), True, (0, 0, 0))
            screen.blit(
                pitch_text, (start_x, screen.get_height() - volume - 20))
            screen.blit(
                volume_text, (start_x, screen.get_height() - volume - 40))


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


def test_visual():
    volume_changes = [
        (3.0, 8.0, 130),  # Louder at 7s, lasting for 3s
        (10.0, 380.0, -100)  # Quieter at 15s, lasting for 3s
    ]

    modified_score = modify_volume('input.mid', volume_changes)
    modified_score.write('midi', 'output_file.mid')

    visualize_notes('input.mid', 'output_file.mid')
    visualize_midi('input.mid', 'output_file.mid')


if __name__ == "__main__":
    test_visual()
