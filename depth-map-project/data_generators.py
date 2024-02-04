import numpy as np
import tensorflow as tf
from keras.utils import Sequence
import cv2
from diode import DIODE
import os
from PIL import Image
from scipy import ndimage


class DiodeDataGenerator(Sequence):
    def __init__(self, split, scene_types, meta_fname, data_root, batch_size, dim, shuffle=True):
        self.batch_size = batch_size
        self.diode = DIODE(splits=split, scene_types=scene_types, meta_fname=meta_fname, data_root=data_root)
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
        # if flip:
        #     x = np.fliplr(x)
        x = tf.image.convert_image_dtype(x, tf.float32)

        return x
    

    def pre_process_depth(self, y, mask, flip):

        # Preprocessing for y
        mask = mask > 0
        max_depth = min(300, np.percentile(y, 99))
        max_depth = max(max_depth, 1)
        y = np.clip(y, 0.1, max_depth)

        # Apply logarithmic transformation and handle inf/nan values
        y = np.log(np.maximum(y, 0.001), where=mask)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.ma.masked_where(~mask, y)
        
        # Clip again after log transformation
        y = np.clip(y, 0.1, np.log(max_depth))

        # Normalize the depth map between 0 and 1
        y_min = y.min()
        y_max = y.max()
        y = (y - y_min) / (y_max - y_min)

        # Resize and expand dimensions
        y = cv2.resize(y, self.output_dim)
        y = np.expand_dims(y, axis=2)
        # if flip:
        #     y = np.fliplr(y)

        y = tf.image.convert_image_dtype(y, tf.float32)
        return y


class RsDataGenerator(Sequence):
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
        """
        Returns the number of batches in the data generator.

        Returns:
            int: Number of batches.

        """
        return int(np.floor(self.data_size / self.batch_size))


    def on_epoch_end(self):
        """
        Shuffles the indexes after each epoch.

        """
        np.random.shuffle(self.indexes)


    def __getitem__(self, index):
        """
        Generates a batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            tuple: A tuple containing the input data and target data.

        """
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
        image_name = f"image_{index:04d}.png"
        depth_name = f"depth_{index:04d}.npy"

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
        x = (x - x.min()) / (x.max() - x.min())

        if flip:
            x = np.fliplr(x)

        x = tf.image.convert_image_dtype(x, tf.float32)

        return x
    
    def pre_process_depth(self, y, flip):

        mask = y == 0

        # For each zero-valued pixel, find the distance to the nearest non-zero pixel
        distances, indices = ndimage.distance_transform_edt(mask, return_indices=True)

        # Map each zero-valued pixel to the value of its nearest non-zero neighbor
        y[mask] = y[tuple(indices[i][mask] for i in range(y.ndim))]
        # temp untill new data
        crop_sides = 190
        crop_vert = 65
        y = y[crop_vert:-crop_vert, crop_sides:-crop_sides-25]

        y = cv2.resize(y, self.output_dim)

        max_depth = min(10, np.percentile(y, 99))
        max_depth = max(max_depth, 1)
        y = np.clip(y, 0.1, max_depth)

        y = (y - y.min()) / (y.max() - y.min())

        y = np.expand_dims(y, axis=2)

        if flip:
            y = np.fliplr(y)

        y = tf.image.convert_image_dtype(y, tf.float32)

        return y

class RsDiodeDataGenerator(Sequence):
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
        