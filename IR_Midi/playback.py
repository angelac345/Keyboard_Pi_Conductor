import midi2audio
import pygame

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

if __name__ == "__main__":
    import sys
    play_audio_file(sys.argv[1])