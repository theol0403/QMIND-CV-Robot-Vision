import numpy as np
import tensorflow as tf
from keras.utils import Sequence
import cv2
from diode import DIODE
import os
from PIL import Image
from scipy import ndimage



class DiodeDataGenerator(Sequence):
    """
    A data generator for the DIODE dataset, providing batches of image, depth, and mask data for training. Utilizes a modified DIODE dataset object from the diode python toolkit package. see references for more information.

    Attributes:
        batch_size (int): Number of samples per batch.
        diode (DIODE): DIODE dataset object.
        data_size (int): Total number of samples in the dataset.
        indexes (np.array): Array of sample indexes, used for shuffling.
        dim (tuple): Dimensions of the input images (height, width).
        output_dim (tuple): Dimensions for the output depth images (height, width), typically half the input dimensions.
        shuffle (bool): Whether to shuffle the data at the start and end of each epoch.

    Parameters:
        split (str): The dataset split to use (e.g., 'train', 'val').
        scene_type (str): The type of scene to use (e.g., 'indoors', 'outdoors'
        meta_fname (str): Path to the metadata file.
        data_root (str): Root directory of the DIODE dataset.
        batch_size (int): Number of samples per batch.
        dim (tuple): Dimensions of the input images (height, width).
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    """
    def __init__(self, split, scene_type, meta_fname, data_root, batch_size, dim, shuffle=True):
        self.batch_size = batch_size
        self.diode = DIODE(splits=split, scene_types=scene_type, meta_fname=meta_fname, data_root=data_root)
        self.data_size = len(self.diode)
        self.indexes = np.arange(self.data_size)
        if shuffle:
            np.random.shuffle(self.indexes)
        self.dim = dim

        # not including channels
        self.output_dim = (self.dim[0]//2, self.dim[1]//2)


    def __len__(self):

        return int(np.ceil(len(self.diode) / float(self.batch_size)))


    def load(self, index):

        x, y, mask = self.diode[index]
        return x, y, mask
    

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


    def __getitem__(self, index):

        x_batch, y_batch = [], []

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        for i in indexes:
            x, y, mask = self.load(i)

            flip = np.random.rand() > 0.5
            x = self.pre_process_image(x, flip)
            y = self.pre_process_depth(y, mask, flip)

            x_batch.append(x)
            y_batch.append(y)
        
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        return x_batch, y_batch
    

    def pre_process_image(self, x, flip):
        x = cv2.resize(x, self.dim[:2])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x / 255.0
        if flip:
            x = np.fliplr(x)
        x = tf.image.convert_image_dtype(x, tf.float32)

        return x
    

    def pre_process_depth(self, y, mask, flip):
        # clip any abnormally high values
        y = np.clip(y, y.min(), np.percentile(y, 99))

        # handle any invalid data
        mask = mask < 1.0

        # For each invalid-valued pixel, find the distance to the nearest non-masked pixel
        _, indices = ndimage.distance_transform_edt(mask, return_indices=True)
        # Map each zero-valued pixel to the value of its nearest non-zero neighbor
        y[mask] = y[tuple(indices[i][mask] for i in range(y.ndim))] 

        if y.max() != y.min():
            y = (y - y.min()) / (y.max() - y.min())

        # Resize and expand dimensions
        y = cv2.resize(y, self.output_dim)
        y = np.expand_dims(y, axis=2)

        # flip horizontally if necessary
        if flip:
            y = np.fliplr(y)

        # convert to float for model
        y = tf.image.convert_image_dtype(y, tf.float32)

        return y


class RsDataGenerator(Sequence):
    """
    A data generator for RealSense data.

    Attributes and parameters similar to `DiodeDataGenerator`, tailored for RealSense data:

    - data_root (str): Root directory containing RealSense image and depth data.
    - image_root (str): Directory containing RealSense images.
    - depth_root (str): Directory containing RealSense depth maps.

    """
    def __init__(self, data_root, batch_size, dim, shuffle=True):
        self.data_root = data_root
        self.batch_size = batch_size
        self.image_root = self.data_root + "rs_image"
        self.depth_root = self.data_root + "rs_depth"
        self.data_size = len([name for name in os.listdir(self.image_root) if os.path.isfile(os.path.join(self.image_root, name))])
        self.indexes = np.arange(self.data_size)
        if shuffle:
            np.random.shuffle(self.indexes)
        self.dim = dim
        self.output_dim = (self.dim[0]//2, self.dim[1]//2)


    def __len__(self):
        return int(np.floor(self.data_size / self.batch_size))


    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


    def __getitem__(self, index):
        x_batch, y_batch = [], []

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        for i in indexes:
            x, y = self.load(i)

            flip = np.random.rand() > 0.5
            x = self.pre_process_image(x, flip)
            y = self.pre_process_depth(y, flip)
            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        return x_batch, y_batch
    

    def load(self, index):
        # Generate filenames for the image and its corresponding depth map
        image_name = f"frame_{index:04d}.png"
        depth_name = f"data_{index:04d}.npy"

        # Full paths to the files
        image_path = os.path.join(self.image_root, image_name)
        depth_path = os.path.join(self.depth_root, depth_name)

        # Load the image and depth data
        try:
            image = Image.open(image_path)
            depth = np.load(depth_path)
        except IOError as e:
            print(f"Error loading files: {e}")
            return None, None

        return np.array(image), depth

    
    def pre_process_image(self, x, flip):
        x = cv2.resize(x, self.dim[:2])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x / 255.0
        if flip:
            x = np.fliplr(x)
        x = tf.image.convert_image_dtype(x, tf.float32)

        return x
    

    def pre_process_depth(self, y, flip):
        mask = y == 0
        # For each zero-valued pixel, find the distance to the nearest non-zero pixel
        _, indices = ndimage.distance_transform_edt(mask, return_indices=True)
        # Map each zero-valued pixel to the value of its nearest non-zero neighbor
        y[mask] = y[tuple(indices[i][mask] for i in range(y.ndim))]

        if y.max() != y.min():
            y = (y - y.min()) / (y.max() - y.min())

        y = cv2.resize(y, self.output_dim)                                                                                      
        y = np.expand_dims(y, axis=2)

        if flip:
            y = np.fliplr(y)

        y = tf.image.convert_image_dtype(y, tf.float32)

        return y


class RsDiodeDataGenerator(Sequence):
    """
    A unified data generator combining DIODE and RealSense datasets.

    Attributes:
        rs_gen (RsDataGenerator): RealSense data generator instance.
        diode_gen (DiodeDataGenerator): DIODE dataset data generator instance.
        batch_size (int): Number of samples per batch.
        data_size (int): Total number of samples across both datasets.

    Parameters:
        data_root (str): Root directory for both DIODE and RealSense data.
        batch_size (int): Number of samples per batch.
        dim (tuple): Dimensions of the input images (height, width).
        diode_meta_name (str, optional): Filename of the DIODE metadata. Defaults to "diode_meta.json".
        diode_split (str, optional): DIODE dataset split to use. Defaults to "train".
        diode_scene_type (str, optional): Type of scenes from DIODE to use. Defaults to "indoors".
    """
    def __init__(self, data_root, batch_size, dim, diode_meta_name="diode_meta.json", diode_split="train", diode_scene_type="indoors"):
        self.rs_gen = RsDataGenerator(data_root, batch_size, dim)
        self.diode_gen = DiodeDataGenerator(diode_split, diode_scene_type, diode_meta_name, data_root, batch_size, dim)

        self.batch_size = batch_size
        self.data_size = self.rs_gen.data_size + self.diode_gen.data_size


    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))


    def on_epoch_end(self):
        self.rs_gen.on_epoch_end()
        self.diode_gen.on_epoch_end()


    def __getitem__(self, index):
        if index < self.rs_gen.__len__():
            return self.rs_gen.__getitem__(index)
        else:
            return self.diode_gen.__getitem__(index - self.rs_gen.__len__())
        