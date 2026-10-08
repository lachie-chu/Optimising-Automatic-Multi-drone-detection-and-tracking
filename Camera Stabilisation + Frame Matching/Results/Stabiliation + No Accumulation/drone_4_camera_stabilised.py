#Author: Lachlan Rochester
#Last updated: 20/01/2026
#Purpose: Takes in a video, npz file (holds camera matrix and distortion) and .pt file (from yolo training) and tracks the drones bounding box and displays it in a graph

from ultralytics import YOLO
import numpy as np
import math
import cv2 as cv
import csv

def dist_to_centre(b, cx_img, cy_img): #Used to determine which ID to track (Only required if multiple drones and you only want to track one)
    x1, y1, x2, y2 = b.xyxy[0]
    cx = (x1 + x2)/2
    cy = (y1 + y2)/2
    return(cx-cx_img)**2 + (cy-cy_img)**2


## Lachlan Chu Updates (Camera Stabilisation) ##
def apply_affine_to_point(point, transform):

    ## Applies a 2x3 affine transformation to a single point.
    x, y = point

    new_x = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]
    new_y = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]

    return new_x, new_y

def combine_affine(A, B):
    
    ## Combines two 2x3 affine transformations.

    A_3x3 = np.vstack([A, [0, 0, 1]])
    B_3x3 = np.vstack([B, [0, 0, 1]])

    combined = A_3x3 @ B_3x3    

    return combined[:2, :]

def estimate_camera_motion(previous_gray, current_gray,
                           previous_points):

    # Track background features from previous frame
    current_points, status, error = cv.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None
    )

    if current_points is None:
        return None, None, None

    # Only keep successfully tracked points
    status = status.ravel()

    good_previous = previous_points[status == 1]
    good_current = current_points[status == 1]

    # Need enough points to estimate movement
    if len(good_previous) < 10:
        return None, None, None

    # Estimate transformation from CURRENT frame back to PREVIOUS frame.
    #
    # This is important:
    # We want to remove camera movement from the current frame.
    transform, inliers = cv.estimateAffinePartial2D(
        good_current,
        good_previous,
        method=cv.RANSAC,
        ransacReprojThreshold=3
    )

    if transform is None:
        return None, None, None

    return transform, good_current.reshape(-1, 1, 2), good_previous.reshape(-1, 1, 2)



def yolo_track(video_filename): #Uses yolo to get bounding box data
    TARGET_ID = None

    # Stores corrected drones positions
    centre_box = []

    model = YOLO('best.pt') #From latest model training


### Opening the video file, so that transformation can be performed on the video before YOLO or camera calibration

##### L.C Before feeding into the YOLO model, we want to open the video file 
    cap = cv.VideoCapture(video_filename)

    if not cap.isOpened():
        print("Could not opened video.")
        return centre_box
    
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print("Video FPS:", fps)
    print("Video resolution:", width, "x", height)

##### L.C Reading the first frame

    ret, previous_frame = cap.read()

    if not ret:
        print("Could not read first frame.")
        cap.release()
        return centre_box

    previous_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)

##### L.C Finding stationary background figures
    # Drone 4 footage, has the drones in the top half of the sky, so we have to block out this half, else the drones themselves might become stationary points
    # , which is bad because they are constantly moving.
    mask = np.zeros_like(previous_gray)

    mask[int(height * 0.45):height, :] = 255

    previous_points = cv.goodFeaturesToTrack(
        previous_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7,
        mask=mask
    )

    if previous_points is None:
        print("Could not find background features.")
        cap.release()
        return centre_box

##### L.C Cumulative camera transformation
    # At frame 0 there is no correction
    cumulative_transform = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)

    frame_number = 0

##### L.C Process the video frame by frame

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        
        current_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        transform, current_points, previous_good_points = \
            estimate_camera_motion(
                previous_gray,
                current_gray,
                previous_points
            )

        if transform is not None:

            # transform maps:
            #
            # current frame → previous frame
            #
            # Therefore accumulate it so that:
            #
            # current frame → original frame
            cumulative_transform = transform
        else:
            print(
                "Warning: could not estimate camera movement "
                "on frame", frame_number
            )

        ## L.C Perform Yolo on current frame

        results = model.track(
            frame, 
            conf = 0.4, 
            classes = 1, 
            persist = True, 
            verbose = False
        ) #Tracking (short video) with a confidence of 0.4, class of 1 (drone, class 0 is bird) and persist means that it is a sequence of frames

        r = results[0]

        ### Select Target Drone
        
        if TARGET_ID is None:

            correct_box_id = None
            smallest_area = float('inf')

            for box in r.boxes:
                if box.id is not None:
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    width = x2-x1
                    height = y2 - y1

                    area = width * height

                    if area<smallest_area:
                        correct_box_id = int(box.id.item())
                        smallest_area = area

            TARGET_ID = correct_box_id
            if TARGET_ID is not None:
                print("Target Drone ID:", TARGET_ID)

        for box in r.boxes: #Goes through each boundary box in each frame

            ## L.C Without this, it could crash, if box.idis None! 
            if box.id is None:
                continue
            
            if int(box.id.item()) == TARGET_ID: #Makes sure its the target ids box
                x1, y1, x2, y2 = box.xyxy[0].tolist() #Gets the xy coordinates

                #Finds the centre of the bounding box
                centre_x = (x1+x2)/2
                centre_y = (y1+y2)/2

                original_point = (centre_x, centre_y)

                ### L.C Camera shake compensation

                corrected_point = apply_affine_to_point(
                    original_point,
                    cumulative_transform
                )

                corrected_x, corrected_y = corrected_point

                centre_box.append((frame_number, corrected_x, corrected_y))
                break

    cap.release()

    print("Frames processed:", frame_number)
    print("Drone positions obtained:", len(centre_box))

    return centre_box



def convert_to_angles(filename, centre_boxes, yaw, pitch): #Converts pixels too zenith and azimuth angles

    pitch = 90-pitch #Zenith

    angles = []

    data = np.load(filename)  #Load up the npz file (There is also a YAML file with the data but i couldnt get it to work properly)
    matrix = data['matrix']
    distortion = data['distortion']

    print(matrix)
    print(distortion)

    for (frame_number, x_cent, y_cent) in centre_boxes: #Cycles through and calculates the xc and yc assuming zc =1.

        centre_points = np.array([[[x_cent,y_cent]]], dtype=np.float32)

        un_dist = cv.undistortPoints(centre_points, matrix, distortion) #Undistorts points and is in 3D camera coordinate system
        

        x,y = un_dist[0,0]

        z = 1 

        #Calculate angles 
        azimuth = yaw + np.rad2deg(math.atan2(x,z))

        zenith = pitch + np.rad2deg(math.atan2(y,z))

        #print(f'Azimuth {azimuth}')
        #print(f'Zenith {zenith}')

        angles.append((frame_number, azimuth, zenith))

    return angles


def write_to_csv(filename,angles): #Writes to csv file

    with open(filename,'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(['Frame', 'Azimuth Angle', 'Zenith Angle'])

        for (frame_number, azimuth, zenith) in angles:

            writer.writerow([frame_number,azimuth,zenith])







def main():

    #Drone 4 (Secondary Camera) (NED - (1.03, -19.9, 0))
    file_video = '../videos/Drone4_30Seconds.mp4'
    filename = '../Calibrations/Drone_Four_Calibration.npz'
    centre_boxes = yolo_track(file_video)

    print("Number of corrected drone positions:", len(centre_boxes))

    drone_four_angles = convert_to_angles(filename, centre_boxes,28.1,19.1)
    write_to_csv('Drone_Four_Angles_Camera_Stabilised.csv', drone_four_angles) 
    

if __name__ == '__main__':
    main()
    

        