import cv2
import copy 
import mediapipe as mp
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt 

from argparse import ArgumentParser
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def get_bounding_box(image, results): 
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
        # Bounding box calculation
        brect = calc_bounding_rect(image, hand_landmarks)
        return brect


def video_to_box_info(vid_path: str) -> list:
    
    video = cv2.VideoCapture(vid_path)

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frame_count/fps 

    info = {'fps': fps, 'duration': duration, 'frame_count': frame_count}    
    # Check if video opened successfully
    if not video.isOpened():
        print("Error opening video file")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    
    progress_bar = tqdm(total=frame_count, desc="Processing frames", ncols=0) 

    bounding_boxes = []
    # Read until video is completed
    curr_frame = 1 
    while video.isOpened():

        ret, image = video.read()
        if not ret:
            break
        image = cv2.flip(image, 1)  # Mirror display

        # Detection implementation #############################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            # brect = get_bounding_box(image, results) 
            thumb = results.multi_hand_landmarks[0].landmark[4] 
            middle = results.multi_hand_landmarks[0].landmark[12]
            brect = (thumb.x - middle.x) ** 2 + (thumb.y - middle.y)**2 + (thumb.z - middle.z) **2
            brect = np.sqrt(brect)

            timestamp = curr_frame / fps 
            
            
            bounding_boxes.append((timestamp, brect))
        curr_frame += 1
        progress_bar.update(1)

    progress_bar.close()
    # When everything done, release the video capture object
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return info, bounding_boxes  



def bbox_to_scale(bounding_boxes): 
    diag_len = [ (time, np.sqrt((box[2] - box[0])**2 + (box[3] - box[1])**2)) for time, box in bounding_boxes]

    timestamps, diag_len = zip(*diag_len)

    diag_len = np.array(diag_len)
    timestamps = np.array(timestamps)

    frame_num = np.arange(len(diag_len)) 

    data = np.hstack((np.expand_dims(timestamps, -1), np.expand_dims(diag_len, -1))) 
    

    # smoothign forward by taking hte mean of hte next `smooth_size` frames, including the current one

    smoothing = copy.deepcopy(data)
    smooth_size = 3
    for i in range(len(data)): 
        smoothing[i, 1] = np.mean(data[i:min(len(data), i+smooth_size), 1]) 


    # mask the data points that have a slope difference smaller than `epsi` with their neighbors
    slope_view = 4
    epsi = 1.1
    data = smoothing

    slope_diffs = []
    for i in range(len(data)):
        if i == 0 or i == len(data) - 1: 
            slope_diffs.append(0)
            continue
        slope_left = np.polyfit(
            data[max(0, i-slope_view):i+1, 0], 
            data[max(0, i-slope_view):i+1, 1], 
            1
        )

        slope_right = np.polyfit(
            data[i:min(len(data), i+slope_view), 0], 
            data[i:min(len(data), i+slope_view), 1], 
            1
        )

        slope_diff = abs(slope_left[0] - slope_right[0])
        slope_diffs.append(slope_diff)


    slope_diffs = np.array(slope_diffs) # contains at each point the difference in slope between points to its left and to its right
    mask = [True] * slope_view 

    for i in range(slope_view, len(slope_diffs) - slope_view): 
        max_diff = np.max(slope_diffs[i-slope_view:i+slope_view])
        
        if max_diff < epsi:
            mask.append(False)
        elif max_diff == slope_diffs[i]: 
            mask.append(True)
        else: 
            mask.append(False)

    mask.extend([True] * slope_view)
    
    masked = data[mask]             # masked contains the data points that have a slope difference greater than `epsi` to left and right 

    plt.xlabel('Frame Number')
    plt.ylabel('BBox Diagonal Length') 
    plt.plot(smoothing[:, 0], smoothing[:, 1], c='b') 
    plt.scatter(masked[:, 0], masked[:, 1], c='r')
    plt.savefig('plot.png')

    
    max_expand, min_expand = diag_len.max(), diag_len.min() 
    
    open_scale = (masked[:, 1] - min_expand) / (max_expand - min_expand)

    return masked[:, 0], open_scale
    

def converter(timestamp, open_scale): 
    assert len(timestamp) == len(open_scale) 

    output = [] 
    for i in range(len(open_scale) - 1): 
        output.append((timestamp[i], timestamp[i+1], (open_scale[i+1] - open_scale[i])*100))
    
    return output 
     
def main_disgarded(video_path): 
    info, bounding_boxes = video_to_box_info(video_path) 
    timestamps, open_scale = bbox_to_scale(bounding_boxes) 
    # timestamps = timestamps / info['fps']
    ir = converter(timestamps, open_scale) 
    return info, ir 

def gesture_ir_main(video_path): 
    info, bounding_boxes = video_to_box_info(video_path)
    diag_len = bounding_boxes
    # diag_len = [ (time, np.sqrt((box[2] - box[0])**2 + (box[3] - box[1])**2)) for time, box in bounding_boxes]


    timestamps, diag_len = zip(*diag_len)
    plt.plot(timestamps, diag_len) 
    plt.savefig('plot.png')
    
    box_sizes = np.array(diag_len) 
    max_expand, min_expand = box_sizes.max(), box_sizes.min() 
    
    open_scale = (box_sizes - min_expand) / (max_expand - min_expand)
    ir = converter(timestamps, open_scale)
    return info, ir

if __name__ == "__main__":

    parser = ArgumentParser() 
    parser.add_argument('--vid_path', type=str)
    args = parser.parse_args() 

    print(gesture_ir_main(args.vid_path))
