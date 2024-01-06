# Standard library imports
import os

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2


# Local application imports
from diode import DIODE, plot_depth_map

# Constants
annotation_folder = "/val/"
split = 'val'
scene_type = 'indoors'
meta_fname = './diode_meta.json'
data_root = '/home/colin/projects/QMIND-CV-Robot-Vision/depth-map-project'
diode = DIODE(splits=split, scene_types=scene_type, meta_fname=meta_fname, data_root=data_root)

# Ensure annotation folder exists
if not os.path.exists(os.path.abspath(".") + annotation_folder):
    annotation_zip = tf.keras.utils.get_file(
        "val.tar.gz",
        cache_subdir=os.path.abspath("."),
        origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
        extract=True,
    )

#define hyperparameters
batch_size = 6
epochs = 1
learning_rate = 0.001
dim = (256, 256)


class CustomDataGenerator(Sequence):
    """
    CustomDataGenerator is a data generator class that generates batches of data for training or evaluation.

    Args:
        splits (list): List of splits to include in the data generator.
        scene_types (list): List of scene types to include in the data generator.
        meta_fname (str): File name of the metadata file.
        data_root (str): Root directory of the data.
        batch_size (int): Number of samples per batch.

    Attributes:
        batch_size (int): Number of samples per batch.
        diode (DIODE): DIODE object for loading and processing data.
        data_size (int): Total number of samples in the data generator.

    Methods:
        __len__(): Returns the number of batches in the data generator.
        __getitem__(index): Returns a batch of data at the given index.
        load(batch_id): Loads and preprocesses a single batch of data.
        data_generation(index): Generates a batch of data.

    """

    def __init__(self, splits, scene_types, meta_fname, data_root, batch_size):
        self.batch_size = batch_size
        self.diode = DIODE(splits=splits, scene_types=scene_types, meta_fname=meta_fname, data_root=data_root)
        self.data_size = len(self.diode)


    def __len__(self):
        """
        Returns the number of batches in the data generator.

        Returns:
            int: Number of batches.

        """
        return int(np.ceil(len(self.diode) / float(self.batch_size)))


    def __getitem__(self, index):
        """
        Returns a batch of data at the given index.

        Args:
            index (int): Index of the batch.

        Returns:
            tuple: A tuple containing the input data and target data.

        """
        return self.data_generation(index)


    def load(self, batch_id):
        """
        Loads and preprocesses a single batch of data.

        Args:
            batch_id (int): Index of the batch.

        Returns:
            tuple: A tuple containing the preprocessed input data, target data, and mask.

        """
        x, y, mask = self.diode[batch_id]

        x = cv2.resize(x, dim)
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))  # Min-Max normalization

        y = cv2.resize(y, dim)
        y = tf.image.convert_image_dtype(y, tf.float32)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))  

        # Ignore invalid pixels in the depth map by applying the mask
        mask = cv2.resize(mask, dim)
        mask = mask > 0
        y = np.ma.masked_where(~mask, y)

        return x, y


    def data_generation(self, index):
        """
        Generates a batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            tuple: A tuple containing the input data and target data.

        """
        x_batch, y_batch = [], []
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        for i in range(start_index, min(end_index + 1, self.data_size)):
            x, y = self.load(i)
            x_batch.append(x)
            y_batch.append(y)
        
        return np.array(x_batch), np.array(y_batch)


class DepthMapModel(tf.keras.Model):
    """
    Model for generating depth maps from input images.
    """

    def __init__(self):
        super(DepthMapModel, self).__init__()

        # Insert class layers here
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(dim[0], dim[1], 3))
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.upsample1 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')  # Output layer for depth map
    
    def call(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Output tensor of shape (batch_size, height, width, 1) representing the depth map.
        """
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.upsample1(x)
        x = self.conv3(x)

        return x


model = DepthMapModel()

model.compile(optimizer='adam', loss='mean_squared_error')

custom_data_generator = CustomDataGenerator(splits=split, scene_types=scene_type, meta_fname=meta_fname, data_root=data_root, batch_size=batch_size)

model.fit(custom_data_generator , epochs=epochs, steps_per_epoch=30)


fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(10, 15))  # Adjust as needed

# Load and predict 5 images
for i in range(5):
    x, y, mask = diode[i]
    x = cv2.resize(x, dim)

    # Ensure x is 4D: (samples, height, width, channels)
    x = np.expand_dims(x, axis=0)

    mask = cv2.resize(mask, dim)
    mask = mask > 0
    y = cv2.resize(y, dim)
    y = np.ma.masked_where(~mask, y)

    # Convert x to float32
    x = x.astype('float32') / 255

    # Predict depth map
    y_pred = model.predict(x)

    # Plot original image, truth depth map, and predicted depth map using matplotlib
    axs[i, 0].imshow(x[0])
    axs[i, 0].set_title('Original Image')
    axs[i, 1].imshow(y)
    axs[i, 1].set_title('Truth Depth Map')
    axs[i, 2].imshow(y_pred[0, :, :, 0])
    axs[i, 2].set_title('Predicted Depth Map')

# Remove axis for all subplots
for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()