import pygame
import PythonPiano.piano_lists as pl
from pygame import mixer
from music21 import note, stream,tempo
from threading import Thread 
import time
class AudioRecorder: 
    def __init__(self,save_path, bpm=90):
        print('initializing new audio recorder')
        self.save_path = save_path
        self.bpm = bpm
        self.metronome_mark = tempo.MetronomeMark(number=self.bpm)
        self.left_oct = 4
        self.right_oct = 5
        self.white_notes = pl.white_notes
        self.black_notes = pl.black_notes
        self.black_labels = pl.black_labels
        # windowSurfaceObj = pygame.display.set_mode((64,48),1,16)

        self.initialize(bpm)
    def initialize(self, bpm=None): 
        if bpm is not None: 
            self.bpm = bpm
            self.metronome_mark = tempo.MetronomeMark(number=self.bpm)        
        self.score = stream.Stream()
        self.score.append(self.metronome_mark)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(50)

        self.white_sounds = []
        self.black_sounds = []
        for i in range(len(self.white_notes)):
            
            self.white_sounds.append(mixer.Sound(f'/home/pi/Documents/ECE5725_final_proj/PythonPiano/assets/notes/{self.white_notes[i]}.wav'))

        for i in range(len(self.black_notes)):
            self.black_sounds.append(mixer.Sound(f'/home/pi/Documents/ECE5725_final_proj/PythonPiano/assets/notes/{self.black_notes[i]}.wav'))


        
        self.note_with_time = [] 
        self.left_dict = {'Z': f'C{self.left_oct}', 'S': f'C#{self.left_oct}', 'X': f'D{self.left_oct}', 'D': f'D#{self.left_oct}',
                          'C': f'E{self.left_oct}', 'V': f'F{self.left_oct}', 'G': f'F#{self.left_oct}', 'B': f'G{self.left_oct}',
                          'H': f'G#{self.left_oct}', 'N': f'A{self.left_oct}', 'J': f'A#{self.left_oct}', 'M': f'B{self.left_oct}'}
        
        self.right_dict = {
            'R': f'C{self.right_oct}', 
            '5': f'C#{self.right_oct}', 
            'T': f'D{self.right_oct}', 
            '6': f'D#{self.right_oct}',
            'Y': f'E{self.right_oct}', 
            'U': f'F{self.right_oct}', 
            '8': f'F#{self.right_oct}', 
            'I': f'G{self.right_oct}',
            '9': f'G#{self.right_oct}', 
            'O': f'A{self.right_oct}', 
            '0': f'A#{self.right_oct}', 
            'P': f'B{self.right_oct}'
        }

    def start_recording(self): 
        self.recording = True 
        self.record()

    def process_event(self, event): 
        if event.type == pygame.QUIT:
            self.clean()
            return
        if event.type == pygame.TEXTINPUT:
            if event.text.upper() in self.left_dict:
                keynote = self.left_dict[event.text.upper()] 
                self.note_with_time.append((keynote, time.time()))

                if self.left_dict[event.text.upper()][1] == '#':
                    index = self.black_labels.index(keynote)

                    self.black_sounds[index].play(0, 1000)
                else:
                    index = self.white_notes.index(keynote)
                    self.white_sounds[index].play(0, 1000)
                
                # self.score.append(note.Note(keynote, type='quarter'))
            if event.text.upper() in self.right_dict:
                keynote = self.right_dict[event.text.upper()] 
                self.note_with_time.append((keynote, time.time()))

                if self.right_dict[event.text.upper()][1] == '#':
                    index = self.black_labels.index(keynote)
                    self.black_sounds[index].play(0, 1000)
                else:
                    index = self.white_notes.index(keynote)
                    self.white_sounds[index].play(0, 1000) 
                
                # self.score.append(note.Note(keynote, type='quarter'))
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                print('enter:stop')
                self.save_recording()
                return

    def record(self):
        i = 0 
        while self.recording:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.clean()
                    return
                if event.type == pygame.TEXTINPUT:
                    if event.text.upper() in self.left_dict:
                        keynote = self.left_dict[event.text.upper()] 

                        if self.left_dict[event.text.upper()][1] == '#':
                            index = self.black_labels.index(keynote)

                            self.black_sounds[index].play(0, 1000)
                        else:
                            index = self.white_notes.index(keynote)
                            self.white_sounds[index].play(0, 1000)
                        
                        self.note_with_time.append((keynote, time.time()))
                        self.score.append(note.Note(keynote, type='quarter'))
                    
                    if event.text.upper() in self.right_dict:
                        keynote = self.right_dict[event.text.upper()] 

                        if self.right_dict[event.text.upper()][1] == '#':
                            index = self.black_labels.index(keynote)
                            self.black_sounds[index].play(0, 1000)
                        else:
                            index = self.white_notes.index(keynote)
                            self.white_sounds[index].play(0, 1000) 
                        
                        self.note_with_time.append((keynote, time.time()))
                        # self.score.append(note.Note(keynote, type='quarter'))
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        print('enter:stop')
                        self.save_recording()
                        return



    def save_recording(self):
        eighth_timestep = 1 / (self.bpm * 2 / 60)
        for i in range(len(self.note_with_time) - 1): 
            curr_note, curr_time = self.note_with_time[i] 
            _, next_time = self.note_with_time[i+1]
            
            n_time_steps = round((next_time - curr_time) / eighth_timestep)
            self.score.append(note.Note(curr_note, quarterLength=n_time_steps/2))
            
        self.score.append(note.Note(self.note_with_time[-1][0], quarterLength=2))

        # breakpoint()
        print('stop record')
        self.recording = False
        self.score.write('midi', fp=self.save_path)
        print(f"Recording stopped. MIDI file saved as {self.save_path}")
        pygame.mixer.quit()

if __name__ == "__main__":
    recorder = AudioRecorder(save_path='test_recorder.mid')
    print('Hit Enter to Start Recording')
    input() 
    recorder.start_recording()
    print('Hit Enter to stop recording')
    input()
    recorder.stop_recording() 