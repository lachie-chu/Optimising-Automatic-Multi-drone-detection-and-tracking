import cv2
import os

video_path = "Videos/Drone2_8_12_2025_1150.MP4" 
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save every frame
    filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()

print("Done")