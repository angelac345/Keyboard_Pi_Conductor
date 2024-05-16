
from GUI import *
import pygame
from pygame.time import Clock
from threading import Thread 
from metronome import Metronome
        
init_system()
gui = GUI(width=800, height=600, fps=2) 


gui.add_text(
    text="Are you ready to produce your music?", 
    center_frac_x=0.5, 
    center_frac_y = 0.25
)

gui.add_circle(radius=50, center_frac_x=0.5, center_frac_y=0.75) 
metronome = Metronome(frame=gui, fps=2)

gui.add_per_frame(metronome.switch)
running = True 

clk = Clock()
while running: 
    for event in pygame.event.get(): 
        exit = gui.event_trigger(event) 
        if exit: 
            running = False
        
    
    gui.reset() 
    gui.render()


metronome.stop() 


