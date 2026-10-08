#Author: Lachlan Rochester
#Last updated: 30/01/2026
#Purpose: Find the drone paths for drone 1 and drone 2 and then plot


from ultralytics import YOLO
import numpy as np
import math
import cv2 as cv
import csv
import matplotlib.pyplot as plt

def test_yolo_ids(video_filename):

    model = YOLO('best.pt')

    results = model.track(
        video_filename,
        conf=0.3,
        classes=1,
        persist=True,
        stream=True,
        verbose=False
    )

    csv_filename = "Drone3_YOLO_IDs.csv"

    frame_number = 0

    with open(csv_filename, 'w', newline='') as file:

        writer = csv.writer(file)

        # CSV headings
        writer.writerow([
            "Frame",
            "Drone 1 YOLO ID",
            "Drone 2 YOLO ID"
        ])

        for r in results:

            frame_number += 1

            drone1_id = "No detection"
            drone2_id = "No detection"

            detections = []

            # Get YOLO IDs from this frame
            for box in r.boxes:

                if box.id is not None:

                    yolo_id = int(box.id.item())
                    detections.append(yolo_id)

            # First detection
            if len(detections) >= 1:
                drone1_id = detections[0]

            # Second detection
            if len(detections) >= 2:
                drone2_id = detections[1]

            # Write one row for the frame
            writer.writerow([
                frame_number,
                drone1_id,
                drone2_id
            ])

    print("YOLO ID TEST")

    print(f"Frames analysed: {frame_number}")
    print(f"CSV file saved as: {csv_filename}")

    return csv_filename


def main():

    video_file = '../videos/Drone_3_1150_30Seconds.mp4'

    test_yolo_ids(video_file)


if __name__ == '__main__':

    main()

