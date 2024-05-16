from music21 import converter, stream, tempo, meter


def seconds_to_offset(seconds, tempo_changes, time_signature):
    """
    Convert seconds to offset (quarter note lengths) based on the given tempo changes and time signature.

    Args:
        seconds (float): Time in seconds.
        tempo_changes (list): List of (offset, bpm) tuples representing tempo changes.
        time_signature (music21.meter.TimeSignature): Time signature of the MIDI file.

    Returns:
        float: Offset in quarter note lengths.
    """
    offset = 0.0
    remaining_seconds = seconds

    for tempo_start_offset, bpm in tempo_changes:
        qpm = bpm / 60
        quarter_duration = 60 / qpm

        # Calculate the time range for the current tempo
        if len(tempo_changes) == 1:
            tempo_end_offset = float('inf')
        else:
            tempo_end_offset = tempo_changes[tempo_changes.index(
                (tempo_start_offset, bpm)) + 1][0]

        # Calculate the offset and remaining time for the current tempo
        tempo_duration = min(
            remaining_seconds, (tempo_end_offset - offset) * quarter_duration)
        offset += tempo_duration / quarter_duration
        remaining_seconds -= tempo_duration

        if remaining_seconds <= 0:
            break

    # Convert quarter notes to offset based on the time signature
    offset /= time_signature.ratioString

    return offset


def modify_volume(midi_file, volume_changes):
    score = converter.parse(midi_file)

    # Get the tempo changes and time signature
    tempo_changes = [
        (0.0, score.flat.getElementsByClass(tempo.MetronomeMark)[0].number)]
    for tempo_event in score.flat.getElementsByClass(tempo.TempoIndication):
        tempo_changes.append(
            (tempo_event.offset, tempo_event.getnaio().getQuarterBPM()))
    time_signature = score.flat.getElementsByClass(meter.TimeSignature)[0]

    flat_score = score.flat

    for start_time_seconds, end_time_seconds, volume_change in volume_changes:
        start_offset = seconds_to_offset(
            start_time_seconds, tempo_changes, time_signature)
        end_offset = seconds_to_offset(
            end_time_seconds, tempo_changes, time_signature)

        for note in flat_score.notes:
            if start_offset <= note.offset < end_offset:
                if volume_change == -100:
                    # If volume_change is -100, set the velocity to 0 (silence)
                    note.volume.velocity = 0
                else:
                    new_velocity = int(
                        note.volume.velocity * (1 + volume_change / 100))
                    # Clamp the velocity between 0 and 127 (MIDI range)
                    note.volume.velocity = max(0, min(127, new_velocity))

    return score


def test_volume_modification():
    volume_changes = [
        (3.0, 8.0, 130),  # Louder at 3s, lasting for 5s
        (10.0, 380.0, -100)  # Quieter at 10s, lasting for infinite time
    ]

    modified_score = modify_volume(
        'input.mid', volume_changes)

    modified_score.write('midi', 'output_file.mid')


if __name__ == "__main__":
    test_volume_modification()
