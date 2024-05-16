white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

import time
from pygame import mixer

class Metronome: 
    def __init__(self, frame, fps): 
        self.frame = frame 
        self.bit = False 

        self.prev_switch = None
        self.fps = fps
        self.time_step = 1 / fps

        self.sound = mixer.Sound(f'/home/pi/Documents/ECE5725_final_proj/PythonPiano/Synth_Block_F_lo.wav')
    
    def reset_fps(self, fps): 
        self.fps = fps 
        self.time_step = 1/fps
        
    def switch(self): 
        
        if self.prev_switch is not None and time.time() - self.prev_switch < self.time_step: 
            return
        
        self.prev_switch = time.time()
        self.bit = not self.bit
        center, radius, color = self.frame.circles[0] 


        if self.bit: 
            self.frame.circles[0] = (center, radius, red) 
            self.sound.play()
        else: 
            self.frame.circles[0] = (center, radius, white) 
