'''
Utility functions for tf.data pipeline
'''

'''
Author: Mark Harvey
'''

import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf


def parser(data_record):
    ''' TFRecord parser '''

    feature_dict = {
      'label' : tf.io.FixedLenFeature([], tf.int64),
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width' : tf.io.FixedLenFeature([], tf.int64),
      'chans' : tf.io.FixedLenFeature([], tf.int64),
      'image' : tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(data_record, feature_dict)
    label = tf.cast(sample['label'], tf.int32)

    h = tf.cast(sample['height'], tf.uint32)
    w = tf.cast(sample['width'], tf.uint32)
    c = tf.cast(sample['chans'], tf.uint32)
    image = tf.io.decode_image(sample['image'], channels=3)
    image = tf.reshape(image,[h,w,c])

    return image, label


def resize_random_crop(x,y,h,w):
    '''
    Image resize & random crop
    Args:     Image and label
    Returns:  augmented image and unchanged label
    '''
    rh = int(h *1.2)
    rw = int(w *1.2)
    x = tf.image.resize(x, (rh,rw), method='bicubic')
    x = tf.image.random_crop(x, [h, w, 3], seed=42)
    return x,y


def resize(x,y,h,w):
    '''
    Image resize
    Args:     Image and label
    Returns:  resized image and unchanged label
    '''
    x = tf.image.resize(x, (h,w), method='bicubic')
    return x,y


def augment(x,y):
    '''
    Image augmentation
    Args:     Image and label
    Returns:  augmented image and unchanged label
    '''
    x = tf.image.random_flip_left_right(x, seed=42)
    x = tf.image.random_brightness(x, 0.1, seed=42)
    x = tf.image.random_contrast(x, 0.9, 1.1, seed=42)
    x = tf.image.random_saturation(x, 0.9, 1.1, seed=42)   
    return x, y


def normalize(x,y):
    '''
    Image normalization
    Args:     Image and label
    Returns:  normalized image and unchanged label
    '''
    # Convert to floating-point & scale to range -1 to +1
    x = (tf.cast(x, tf.float32) * (1. / 127.5)) - 1.0
    return x, y



def input_fn_trn(tfrec_dir,batchsize,height,width):
    '''
    Dataset creation and augmentation for training
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/train_*.tfrecord'.format(tfrec_dir), shuffle=True)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x,y: resize_random_crop(x,y,h=height,w=width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset


def input_fn_test(tfrec_dir,batchsize,height,width):
    '''
    Dataset creation and augmentation for test
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/test_*.tfrecord'.format(tfrec_dir), shuffle=False)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x,y: resize(x,y,h=height,w=width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


