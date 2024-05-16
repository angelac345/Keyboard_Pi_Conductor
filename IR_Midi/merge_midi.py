from music21 import converter, instrument, note, chord, stream


def merge_midi(path_1, path2, out_path):
    """
    Merge two tracks into 1.

    Args:
        path_1, paht_2 (str): Path to the MIDI file.
        out_path (str): Path to the output MIDI file

    Returns:
        music21.stream.Stream: Modified MIDI score.
    """
    midi_file1 = converter.parse(path_1)
    midi_file2 = converter.parse(path2)

    merged_stream = stream.Stream()

    for element in midi_file1.flat:
        merged_stream.append(element)

    for element in midi_file2.flat:
        merged_stream.append(element)

    merged_stream.write("midi", fp=out_path)


if __name__ == "__main__":
    path_1 = "/home/pi/Documents/ECE5725_final_proj/testrun1_final_midi.mid"
    path_2 = "/home/pi/Documents/ECE5725_final_proj/IR_Midi/input.mid"
    out = "/home/pi/Documents/ECE5725_final_proj/IR_Midi/merged_output.mid"
    merge_midi(path_1, path_2, out)
