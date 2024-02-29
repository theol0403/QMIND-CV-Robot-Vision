import cv2
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2
import threading
import numpy as np
from ultralytics import YOLO

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self.update_frame, args=()).start()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame = frame

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


def loss_function(y_true, y_pred):
    depth_loss = tf.reduce_mean(K.abs(y_pred - y_true), axis=-1)
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    grad_loss = tf.reduce_mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
    ssim_loss = tf.clip_by_value((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)
    depth_loss_weight = 0.1
    return ssim_loss + tf.reduce_mean(grad_loss) + depth_loss_weight * tf.reduce_mean(depth_loss)


def pre_process_image(x):

    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x / 255.0
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = np.expand_dims(x, axis=0)

    return x


def reshape_image(x):
    h, w = x.shape[:2]
    start = abs(h-w) // 2

    if h > w:
        x = x[start:start+w, :]
    else:
        x = x[:, start:start+h]

    x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
    x = cv2.resize(x, (256, 256))

    return x


# Function to load YOLO model
def load_yolo():
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.fuse() # does this actually make it faster?
    return yolo_model


def run_yolo(model, old_image, image):
    # Create a copy of the image
    image_copy = image.copy()

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
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # class name
            cls = int(box.cls[0])

            # add classification
            cv2.putText(image_copy, classNames[cls], [x1,y1], cv2.FONT_HERSHEY_SIMPLEX,1, (50,205,50), 2)

    return image_copy


model = tf.keras.models.load_model('QMIND_depth_model', custom_objects={'loss_function': loss_function})

droidcam_url = "http://192.168.2.34:4747/video"
cam = Camera(droidcam_url)
cam.start()
yolo_model = load_yolo()


while True:
    frame = cam.get_frame()
    if frame is not None:

        frame = reshape_image(frame)

        depth = pre_process_image(frame)
        depth = model.predict(depth, verbose=0)
        depth = np.squeeze(depth)
        depth = (depth * 255).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)      

        frame = cv2.resize(frame, (500,500))
        depth = cv2.resize(depth, (500,500))
        yolo_results = run_yolo(yolo_model, frame, depth)

        cv2.imshow('Webcam Feed', frame)
        cv2.imshow('Depth Map', depth)
        cv2.imshow('YOLO Results', yolo_results)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cam.stop()
cv2.destroyAllWindows()