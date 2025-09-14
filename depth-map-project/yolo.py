import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Function to load YOLO model
def load_yolo():
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.fuse() # does this actually make it faster?
    return yolo_model

# Function to overlay YOLO bounding boxes onto MiDaS image
def run_yolo(model ,old_image, image):
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    results = model(old_image, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # overlay box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # class name
            cls = int(box.cls[0])

            # add classification
            cv2.putText(image, classNames[cls], [x1,y1], cv2.FONT_HERSHEY_SIMPLEX,1, (50,205,50), 2)

    return image

def main():

    # Load YOLO model
    yolo_model = load_yolo()

    # Open webcam
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        # ret, original_image = cap.read()
        original_image = cv2.imread("Screenshot 2024-02-27 at 11.26.55 AM.png")

        # Perform  depth prediction
        depth_map = *****;

        # Perform YOLO object detection
        yolo_results = run_yolo(yolo_model, original_image, depth_map)


        # Display the result
        cv2.imshow('Overlay Image', yolo_results)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()