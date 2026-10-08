#Author: Lachlan Rochester
#Last updated: 30/01/2026
#Purpose: Find the drone paths for drone 1 and drone 2 and then plot


from ultralytics import YOLO
import numpy as np
import math
import cv2 as cv
import csv
import matplotlib.pyplot as plt

def create_kalman(x, y):

    kalman = cv.KalmanFilter(4, 2)

    # State = [x, y, vx, vy]
    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # We only measure x and y
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    # How much we expect the drone's movement to vary
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    # How noisy we expect the YOLO measurements to be
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

    # Initial position
    kalman.statePost = np.array([
        [x],
        [y],
        [0],
        [0]
    ], dtype=np.float32)

    return kalman


def yolo_track(video_filename):

    # Maximum distance in pixels for associating
    # a new YOLO detection with a predicted track
    MAX_DISTANCE = 100

    # Number of frames we allow a drone to be missing
    MAX_MISSED_FRAMES = 100

    model = YOLO('best.pt')

    results = model.track(
        video_filename,
        conf=0.3,
        classes=1,
        persist=True,
        stream=True
    )

    # Our persistent tracks
    tracks = {}

    # Store positions for the rest of your program
    centre_box_one = []
    centre_box_two = []

    tracking_history = []

    next_track_id = 1

    for frame_number, r in enumerate(results):

        # 1. Get YOLO detections

        detections = []

        for box in r.boxes:

            if box.id is not None:

                yolo_id = int(box.id.item())

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                centre_x = (x1 + x2) / 2
                centre_y = (y1 + y2) / 2

                detections.append({
                    'yolo_id': yolo_id,
                    'position': np.array(
                        [centre_x, centre_y],
                        dtype=np.float32
                    )
                })

        # 2. Predict where each existing track should be

        predictions = {}

        for track_id, track in tracks.items():

            prediction = track['kalman'].predict()

            predicted_x = float(prediction[0, 0])
            predicted_y = float(prediction[1, 0])

            predictions[track_id] = np.array(
                [predicted_x, predicted_y],
                dtype=np.float32
            )

        # 3. Match YOLO detections to existing tracks

        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(tracks.keys())

        matches = []

        # Find closest detection to each track
        possible_matches = []

        for track_id in unmatched_tracks:

            predicted_position = predictions[track_id]

            for detection_index in unmatched_detections:

                detection_position = detections[detection_index]['position']

                distance = np.linalg.norm(
                    predicted_position - detection_position
                )

                if distance <= MAX_DISTANCE:

                    possible_matches.append(
                        (distance, track_id, detection_index)
                    )

        # Closest matches first
        possible_matches.sort(key=lambda x: x[0])

        used_tracks = set()
        used_detections = set()

        for distance, track_id, detection_index in possible_matches:

            if track_id in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            matches.append(
                (track_id, detection_index, distance)
            )

            used_tracks.add(track_id)
            used_detections.add(detection_index)

        # 4. Correct Kalman filter for matched tracks

        for track_id, detection_index, distance in matches:
            detection = detections[detection_index]

            old_yolo_id = tracks[track_id]['yolo_id']
            new_yolo_id = detection['yolo_id']

            # Kalman prediction BEFORE correction
            predicted_position = predictions[track_id]

            predicted_x = float(predicted_position[0])
            predicted_y = float(predicted_position[1])

            # Actual YOLO position
            actual_x = float(detection['position'][0])
            actual_y = float(detection['position'][1])

            # Calculate prediction error
            prediction_error = math.sqrt(
                (actual_x - predicted_x) ** 2 +
                (actual_y - predicted_y) ** 2
            )
            measurement = np.array([
                [detection['position'][0]],
                [detection['position'][1]]
            ], dtype=np.float32)

            missed_frames_before_detection = tracks[track_id]['missed_frames']

            # Correct Kalman Filter
            tracks[track_id]['kalman'].correct(measurement)

            tracks[track_id]['position'] = detection['position']

            tracks[track_id]['yolo_id'] = new_yolo_id

            tracks[track_id]['missed_frames'] = 0

            # Determine what happened
            if old_yolo_id == new_yolo_id:
                status = "Detection"

            else:
                status = "Re-associated"

            tracking_history.append({
                'frame': frame_number + 1,
                'track_id': track_id,
                'yolo_id': new_yolo_id,
                'status': status,
                
                # Kalman prediction
                'kalman_x': predicted_x,
                'kalman_y': predicted_y,

                # Actual YOLO detection
                'yolo_x': actual_x,
                'yolo_y': actual_y,

                # Error between prediction and detection
                'prediction_error': prediction_error,

                # Number of missed frames before detection returned
                'missed_frames': missed_frames_before_detection,

                # Existing association distance
                'distance': distance
            })

        # 5. Handle tracks with no detection

        for track_id in list(tracks.keys()):

            if track_id not in used_tracks:

                tracks[track_id]['missed_frames'] += 1

                # Kalman prediction
                predicted_position = predictions[track_id]

                predicted_x = float(predicted_position[0])
                predicted_y = float(predicted_position[1])

                tracks[track_id]['position'] = predictions[track_id]

                tracking_history.append({
                    'frame': frame_number + 1,
                    'track_id': track_id,
                    'yolo_id': "No detection",
                    'status': "Prediction",

                    # Kalman prediction
                    'kalman_x': predicted_x,
                    'kalman_y': predicted_y,

                    # No YOLO measurement
                    'yolo_x': "",
                    'yolo_y': "",

                    # Cannot calculate error without YOLO
                    'prediction_error': "",

                    'missed_frames': tracks[track_id]['missed_frames'],

                    'distance': ""
        
                })

        # 6. Create new tracks for unmatched detections

        for detection_index, detection in enumerate(detections):

            if detection_index not in used_detections:

                if len(tracks) < 2:

                    new_track_id = next_track_id
                    next_track_id += 1

                    x = detection['position'][0]
                    y = detection['position'][1]

                    tracks[new_track_id] = {
                        'kalman': create_kalman(x, y),
                        'position': detection['position'],
                        'yolo_id': detection['yolo_id'],
                        'missed_frames': 0
                    }

                    tracking_history.append({
                        'frame': frame_number + 1,
                        'track_id': new_track_id,
                        'yolo_id': detection['yolo_id'],
                        'status': "New track",

                        # Initial Kalman position
                        'kalman_x': float(detection['position'][0]),
                        'kalman_y': float(detection['position'][1]),

                        # Initial YOLO detection
                        'yolo_x': float(detection['position'][0]),
                        'yolo_y': float(detection['position'][1]),

                        # No prediction has occurred yet
                        'prediction_error': "",

                        # New track has not missed any frames
                        'missed_frames': 0,

                        'distance': ""
                    })

        # 7. Remove tracks missing for too long

        for track_id in list(tracks.keys()):

            if tracks[track_id]['missed_frames'] > MAX_MISSED_FRAMES:

                del tracks[track_id]

        # 8. Save positions for Track 1 and Track 2

        pos_one = None
        pos_two = None

        if 1 in tracks:
            pos_one = tuple(tracks[1]['position'])

        if 2 in tracks:
            pos_two = tuple(tracks[2]['position'])

        if pos_one is not None:
            centre_box_one.append(pos_one)

        if pos_two is not None:
            centre_box_two.append(pos_two)

    return centre_box_one, centre_box_two, tracking_history

def write_tracking_csv(tracking_history, filename):

    with open(filename, 'w', newline='') as file:

        writer = csv.writer(file)

        writer.writerow([
            'Frame',
            'Track ID',
            'YOLO ID',
            'Status',
            'Kalman X',
            'Kalman Y',
            'YOLO X',
            'YOLO Y',
            'Prediction Error',
            'Missed Frames',
            'Association Distance'
        ])

        for row in tracking_history:

            writer.writerow([
                row['frame'],
                row['track_id'],
                row['yolo_id'],
                row['status'],
                row['kalman_x'],
                row['kalman_y'],
                row['yolo_x'],
                row['yolo_y'],
                row['prediction_error'],
                row['missed_frames'],
                row['distance']
            ])


def main():

    #Camera Three (Home)
    video_file = '../videos/Drone_3_1150_30Seconds.mp4' # Needs Video
    box_one, box_two, tracking_history = yolo_track(video_file) # Needs best.pt
    
    write_tracking_csv(tracking_history, 'Drone3_tracking_results.csv')

    #Camera Four (Second)
    #video_file = '../videos/Drone_4_1150_30Seconds.mp4'
    #box_one, box_two, tracking_history = yolo_track(video_file)

    #write_tracking_csv(tracking_history, 'Drone4_tracking_results.csv')


if __name__ == '__main__':

    main()

