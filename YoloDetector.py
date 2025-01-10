import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from Sort import *
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load Video
cap = cv2.VideoCapture("Files/Videos/cars.mp4")
model = YOLO("Weights/yolov8n")

# Load Image
# model = YOLO('Weights/yolov8n.pt')
# results = model("Files/Images/people.jpg", show=True)
# cv2.waitKey(0)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Tracking - Deep Sort
tracker = DeepSort(max_age=50,
                   n_init=10,
                   nms_max_overlap=1.0,
                   max_cosine_distance=0.3,
                   nn_budget=None,
                   override_track_class=None,
                   embedder="mobilenet",
                   half=True,
                   bgr=True,
                   embedder_gpu=True,
                   embedder_model_name=None,
                   embedder_wts=None,
                   polygon=False,
                   today=None)

while True:
    success, img = cap.read()

    results = model(img, stream=True)

    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = int(box.cls[0]) # class label
            currentClass = classNames[cls]

            boundingBox = (x1, y1, w, h)

            # Detect vehicles
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" \
                    and conf > 0.3:
                cvzone.cornerRect(img, boundingBox, l=9, rt=2, colorR=(255, 0, 0))  # Draw bounding box
                currentArray = np.array([x1, y1, x2, y2, conf])
                detection = ([x1, y1, w, h], conf, cls)
                detections.append(detection)

            # Detect people
            if currentClass == "person" and conf > 0.3:
                #cvzone.cornerRect(img, boundingBox, l=9, rt=5)
                cvzone.cornerRect(img, boundingBox, l=9, rt=2, colorR=(255, 0, 0))  # Draw bounding box
                boundingBoxPosition = (max(0, x1), max(35, y1))
                currentArray = np.array([x1, y1, x2, y2, conf])
                detection = ([x1, y1, w, h], conf, cls)
                detections.append(detection)

    resultsTracker = tracker.update_tracks(detections, frame=img)

    for result in resultsTracker:
        # Get bounding box coordinates
        x1, y1, x2, y2 = result.to_tlbr()
        # Get tracker ID
        tracker_id = result.track_id
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        boundingBox = x1, y1, w, h
        # Draw bounding box
        cvzone.cornerRect(img, boundingBox, l=9, rt=2, colorR=(255, 0, 0))
        boundingBoxPosition = (max(0, x1), max(35, y1))
        cvzone.putTextRect(img, f'{int(tracker_id)}', boundingBoxPosition, 1, 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
