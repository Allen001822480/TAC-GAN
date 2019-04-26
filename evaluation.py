#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:29:35 2019

@author: osamuyanagano
"""
import tensorflow as tf
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from scipy import misc
from glob import glob

def inception_score() :
    filenames = glob('./Data/images_generated_from_text/0/*.jpg')
    print(filenames)
    image = get_images(filenames[0])
    print(image.shape)
    image = np.transpose(image, axes=[2, 0, 1])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [3, None, None])

    logits = inception_logits(inception_images)

    IS = get_inception_score(BATCH_SIZE, image, inception_images, logits, splits=10)

    print()
    print("IS : ", IS)

def inception_logits(images, num_splits = 1):
    tfgan = tf.contrib.gan
    print(images.shape)
    images = tf.transpose(images, [1, 2, 0])
    size = 299
    images = tf.image.resize_bilinear(images, size)
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    logits = functional_ops.map_fn(
        fn = functools.partial(tfgan.eval.run_inception, output_tensor = 'logits:0'),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

def get_inception_probs(batch_size, images, inception_images, logits):
    n_batches = len(images) // batch_size
    preds = np.zeros([n_batches * batch_size, 1000], dtype = np.float32)
    for i in range(n_batches):
        inp = images/ 255. * 2 - 1
        preds[i * batch_size:(i + 1) * batch_size] = logits.eval({inception_images:inp})[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(batch_size, images, inception_images, logits, splits=10):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 3)
    assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    print('Calculating Inception Score with 1 image in %i splits' % (splits))
    start_time=time.time()
    preds = get_inception_probs(batch_size, images, logits, inception_images)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits.

def get_images(filename):
    x = misc.imread(filename)
    x = misc.imresize(x, size=[299, 299])
    return x

inception_score()