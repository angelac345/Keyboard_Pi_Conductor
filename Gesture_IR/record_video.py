import cv2 
import numpy as np 
from threading import Thread
import time 
# This is a script that records a video using the camera


class VideoRecorder: 
    def __init__(self, video_path, bpm): 

        self.video_path = video_path
        self.initialize

    def initialize(self, bpm): 
        # Create a VideoCapture object
        self.camera = cv2.VideoCapture(0)
        self.camera_open = False 

        # Check if the camera is opened
        if not self.camera.isOpened(): 
            print("Cannot open camera")
            exit()

        self.bpm = bpm
        self.fps = self.bpm / 60
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (640, 480))

    def spawn_camera(self): 
        self.recording = False 
        self.camera_open = True
        self.record_thread = Thread(target=self.record)
        self.record_thread.start()
        self.timestamp = time.time()

    def start_recording(self): 
        while not self.camera.isOpened(): 
            time.sleep(0.01)
        self.recording = True
        self.timestamp = time.time()

    def stop_recording(self): 
        # Release everything if job is finished

        self.recording = False 
        self.camera_open = False 

        self.record_thread.join() 
        self.camera.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

    def record(self): 
        frames = [] 

        timestep = 1 / self.fps 
        while self.camera_open and self.camera.isOpened(): 
            ret, frame = self.camera.read()

            if not ret: 
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Flip the frame horizontally
            # frame = cv2.flip(frame, 1)
            frame = cv2.flip(frame, 0)

            if self.recording and time.time() - self.timestamp > timestep: 
                self.timestamp = time.time()
                print(f'Appending Frame at {self.timestamp}')
                frames.append(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            k = cv2.waitKey(1) & 0xFF
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        for f in frames: 
            self.video_writer.write(f)
        

if __name__ == "__main__": 
    recorder = VideoRecorder(video_path='output.mp4', bpm=90) 
    input()
    recorder.spawn_camera()
    print('camera spawned')
    input() 
    recorder.start_recording() 
    print('starting recording')
    input() 
    recorder.stop_recording() 
    