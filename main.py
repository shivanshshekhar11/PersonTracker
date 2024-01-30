#Tutorial by ComputerVisionEngineer on YouTube

import os
import cv2
import random
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8n.pt')
tracker = Tracker()
threshold = 0.5

video_path = os.path.join('.','stream','test-01.mp4')
cap = cv2.VideoCapture(video_path)

colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in range(50)]

ret, frame = cap.read()

video_out_path = os.path.join('.','stream','output-01.mp4')
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1],frame.shape[0]))

while ret:

    results = model(frame)

    for result in results:
        detections = []

        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = r

            if int(class_id) == 0 and confidence > threshold:
                detections.append([int(x1), int(y1), int(x2), int(y2), confidence])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), colors[track_id % len(colors)], 3)

    #cv2.imshow('output',frame)
    #cv2.waitKey(5)

    cap_out.write(frame)

    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()