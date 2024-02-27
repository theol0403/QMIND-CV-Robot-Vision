import os.path as osp
from itertools import chain
import json
import os
import numpy as np
from PIL import Image
import tensorflow as tf


_VALID_SPLITS = ('train', 'val', 'test')
_VALID_SCENE_TYPES = ('indoors', 'outdoor')


def check_and_tuplize_tokens(tokens, valid_tokens):
    if not isinstance(tokens, (tuple, list)):
        tokens = (tokens, )
    for split in tokens:
        assert split in valid_tokens
    return tokens


def enumerate_paths(src):
    '''flatten out a nested dictionary into an iterable
    DIODE metadata is a nested dictionary;
    One could easily query a particular scene and scan, but sequentially
    enumerating files in a nested dictionary is troublesome. This function
    recursively traces out and aggregates the leaves of a tree.
    '''
    if isinstance(src, list):
        return src
    elif isinstance(src, dict):
        acc = []
        for k, v in src.items():
            _sub_paths = enumerate_paths(v)
            _sub_paths = list(map(lambda x: osp.join(k, x), _sub_paths))
            acc.append(_sub_paths)
        return list(chain.from_iterable(acc))
    else:
        raise ValueError('do not accept data type {}'.format(type(src)))


class DIODE:

    def __init__(self, meta_fname, data_root, splits, scene_types):
        self.data_root = data_root
        self.splits = check_and_tuplize_tokens(
            splits, _VALID_SPLITS
        )
        self.scene_types = check_and_tuplize_tokens(
            scene_types, _VALID_SCENE_TYPES
        )
        with open(meta_fname, 'r') as f:
            self.meta = json.load(f)

        imgs = []
        for split in self.splits:
            for scene_type in self.scene_types:
                _curr = enumerate_paths(self.meta[split][scene_type])
                _curr = map(lambda x: os.path.join(split, scene_type, x), _curr)
                imgs.extend(list(_curr))
        self.imgs = imgs

    def _load_image(self, path):
        return np.array(Image.open(path))

    def _load_data(self, path):
        return np.load(path).squeeze()

    def _generator(self):
        for im in self.imgs:
            im_fname = os.path.join(self.data_root, '{}.png'.format(im))
            de_fname = os.path.join(self.data_root, '{}_depth.npy'.format(im))
            de_mask_fname = os.path.join(self.data_root, '{}_depth_mask.npy'.format(im))

            yield self._load_image(im_fname), self._load_data(de_fname), self._load_data(de_mask_fname)

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool)
            )
        )
    def __getitem__(self, index):
        if index < 0 or index >= len(self.imgs):
            raise IndexError('Index out of range')
        im = self.imgs[index]
        im_fname = os.path.join(self.data_root, '{}.png'.format(im))
        de_fname = os.path.join(self.data_root, '{}_depth.npy'.format(im))
        de_mask_fname = os.path.join(self.data_root, '{}_depth_mask.npy'.format(im))
        return self._load_image(im_fname), self._load_data(de_fname), self._load_data(de_mask_fname)

    def __len__(self):
        return len(self.imgs)