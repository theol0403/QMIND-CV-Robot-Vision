import cv2
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2
import threading

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

model = tf.keras.models.load_model('QMIND_depth_model', custom_objects={'loss_function': loss_function})

def pre_process_image(x):
    h, w = x.shape[:2]
    start = abs(h-w) // 2

    if h > w:
        x = x[start:start+w, :]
    else:
        x = x[:, start:start+h]

    x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
    x = cv2.resize(x, (256, 256))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x / 255.0
    x = tf.image.convert_image_dtype(x, tf.float32)
    
    return x

droidcam_url = "http://192.168.2.34:4747/video"
cam = Camera(droidcam_url)
cam.start()

while True:
    frame = cam.get_frame()
    if frame is not None:

        frame = pre_process_image(frame)
        frame = np.expand_dims(frame, axis=0)

        depth = model.predict(frame, verbose=0)
        depth = np.squeeze(depth)
        depth = cv2.resize(depth, (640, 480))
        depth_scaled = (depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

        cv2.imshow('Webcam Feed', depth_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.stop()
cv2.destroyAllWindows()