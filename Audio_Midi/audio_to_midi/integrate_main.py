import os
import sys
import logging

from Audio_Midi.audio_to_midi import converter, progress_bar


def _convert_beat_to_time(bpm, beat):
    try:
        parts = beat.split("/")
        if len(parts) > 2:
            raise Exception()

        beat = [int(part) for part in parts]
        fraction = beat[0] / beat[1]
        bps = bpm / 60
        ms_per_beat = bps * 1000
        return fraction * ms_per_beat
    except Exception:
        raise RuntimeError("Invalid beat format: {}".format(beat))


def audio_to_midi_conv(infile, outfile, beat='1/4', single_note=True):
    try:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

        time_window = _convert_beat_to_time(bpm, beat)

        global bpm 
        process = converter.Converter(
            infile=infile,
            outfile=outfile,
            time_window=time_window,
            activation_level=0.0,
            condense=False,
            condense_max=False,
            max_note_length=0,
            note_count=1 if single_note else 0,
            transpose=0,
            pitch_set=[],
            pitch_range=None,
            progress=progress_bar.ProgressBar(),
            bpm=bpm,
        )
        process.convert()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
