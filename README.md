# Optimising-Automatic-Multi-drone-detection-and-tracking
The Repository contains three files:

- Datasets for Complex Background and Occlusion
-> This includes the following list of files:
    - Extracted Footage from Camera
    - Old Dataset
    - Roboflow Obtained (best)
    - yolo_detection.py
    - LatestModel.py

Instructions: 
To run the model, you need open the LatestModel in Kaggle and use the roboflow URL found in the READMEs of each testing to obtain the trained model. This will provide a best.pt, which can then be used with the yolo_detection.py to run and obtain the runs/footage of the detection. Each testing has a README that contains the URL for their respective dataset, the "Extracted Footage from Camera" contains four .py codes in it to obtain the data from footage. You must first perform video to frame for the video, then obtain the annotations through the usage of the yolo_detection.py, where its save_txt is set to True. You then will have to rename the annotations to match the images and finally delete any images without an annotation.

- Camera Stabilisation and Frame Matching
-> This includes the following list of files:
    - Calibrations
    - Results
    - Theoretical
    - YOLO tracking model + Camera stabilisation + Frame matching
    - YOLO tracking model + Frame Matching
    - YOLO tracking model Original

Instructions: 
For camera stabilisation there is two testing, the first is to see the affects of frame matching, and the second is to see the affects of camera stabilisation. To run the frame matching, first run the Single_Drone_Tracking.py and Errors_and_Graphing.py. This will obtain the Camera_vs_Theoretical which shows the comparison between the position of the drones from the camera and the GPS position. For the Camera stabilisation and frame matching the same code is ran in the respective file.

- Maintaining Consistent ID Tracking
-> This includes the following list of files:
    - ID Switches Code
    - Kal + Reassociation
    - videos
    - best.pt
 
Instructions: 
The code for ID Switches is "MDT_IDSwitchCount.py" which once run produces a excel datasheet containing the YOLO IDs for each drone and the frame associated with it. If no detections are found it will write "No detections". This was compared with the results found from the Kal + Reassociation file, which has the code "MDT_Kal+ID.py" which provides the persistent ID found through the new system.
