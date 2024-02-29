import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_generators import DiodeDataGenerator, RsDiodeDataGenerator, RsDataGenerator
import tensorflow.keras.backend as K



def loss_function(y_true, y_pred):
  
  #Cosine distance loss
  depth_loss = tf.reduce_mean(K.abs(y_pred - y_true), axis=-1)
  
  # edge loss for sharp edges
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  grad_loss = tf.reduce_mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
  
  # structural similarity loss assuming target range is 1
  ssim_loss = tf.clip_by_value((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)

  # weightage
  depth_loss_weight = 0.1
  return ssim_loss + tf.reduce_mean(grad_loss) + depth_loss_weight * tf.reduce_mean(depth_loss)

# Load the model
model2 = tf.keras.models.load_model('QMIND_depth_model', custom_objects={'loss_function': loss_function})


def pre_process_image(x):
    x = cv2.resize(x, (256, 256))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x / 255.0
    x = tf.image.convert_image_dtype(x, tf.float32)

    return x


cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = pre_process_image(frame)

    frame = np.expand_dims(frame, axis=0)

    depth = model2.predict(frame)

    depth = np.squeeze(depth)

    depth = cv2.resize(depth, (32,32))

    # Display the resulting frame
    cv2.imshow('Webcam Feed', depth)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break