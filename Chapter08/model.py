#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 06:12:10 2018

@author: zqwu
"""

import deepchem as dc
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from deepchem.data import NumpyDataset, pad_features
from deepchem.metrics import to_one_hot
from deepchem.trans import undo_transforms
from deepchem.data.data_loader import ImageLoader
from sklearn.metrics import confusion_matrix, accuracy_score


class DRModel(dc.models.KerasModel):

  def __init__(self,
               n_tasks=1,
               image_size=512,
               n_downsample=6,
               n_init_kernel=16,
               n_fully_connected=[1024],
               n_classes=5,
               augment=False,
               batch_size=100,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    image_size: int
      Resolution of the input images(square)
    n_downsample: int
      Downsample ratio in power of 2
    n_init_kernel: int
      Kernel size for the first convolutional layer
    n_fully_connected: list of int
      Shape of FC layers after convolutions
    n_classes: int
      Number of classes to predict (only used in classification mode)
    augment: bool
      If to use data augmentation
    """
    self.n_tasks = n_tasks
    self.image_size = image_size
    self.n_downsample = n_downsample
    self.n_init_kernel = n_init_kernel
    self.n_fully_connected = n_fully_connected
    self.n_classes = n_classes
    self.augment = augment

    # inputs placeholder
    self.inputs = tf.keras.Input(
        shape=(self.image_size, self.image_size, 3), dtype=tf.float32)
    # data preprocessing and augmentation
    in_layer = DRAugment(
        self.augment,
        batch_size,
        size=(self.image_size, self.image_size))(self.inputs)
    # first conv layer
    in_layer = layers.Conv2D(int(self.n_init_kernel), kernel_size=7, padding='same')(in_layer)
    in_layer = layers.BatchNormalization()(in_layer)
    in_layer = layers.ReLU()(in_layer)

    # downsample by max pooling
    res_in = layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2))(in_layer)

    for ct_module in range(self.n_downsample - 1):
      # each module is a residual convolutional block
      # followed by a convolutional downsample layer
      in_layer = layers.Conv2D(int(self.n_init_kernel * 2**(ct_module - 1)), kernel_size=1, padding='same')(res_in)
      in_layer = layers.BatchNormalization()(in_layer)
      in_layer = layers.ReLU()(in_layer)
      in_layer = layers.Conv2D(int(self.n_init_kernel * 2**(ct_module - 1)), kernel_size=3, padding='same')(in_layer)
      in_layer = layers.BatchNormalization()(in_layer)
      in_layer = layers.ReLU()(in_layer)
      in_layer = layers.Conv2D(int(self.n_init_kernel * 2**ct_module), kernel_size=1, padding='same')(in_layer)
      res_a = layers.BatchNormalization()(in_layer)

      res_out = res_in + res_a
      res_in = layers.Conv2D(
          int(self.n_init_kernel * 2**(ct_module + 1)),
          kernel_size=3,
          strides=2,
          activation=tf.nn.relu,
          padding='same')(res_out)
      res_in = layers.BatchNormalization()(res_in)

    # max pooling over the final outcome
    in_layer = layers.Lambda(lambda x: tf.reduce_max(x, axis=(1,2)))(res_in)

    regularizer = tf.keras.regularizers.l2(0.1)
    for layer_size in self.n_fully_connected:
      # fully connected layers
      in_layer = layers.Dense(layer_size, activation=tf.nn.relu, kernel_regularizer=regularizer)(in_layer)
      # dropout for dense layers
      #in_layer = layers.Dropout(0.25)(in_layer)

    logit_pred = layers.Dense(self.n_tasks * self.n_classes)(in_layer)
    logit_pred = layers.Reshape((self.n_tasks, self.n_classes))(logit_pred)
    output = layers.Softmax()(logit_pred)

    keras_model = tf.keras.Model(inputs=self.inputs, outputs=[output, logit_pred])
    super(DRModel, self).__init__(
        keras_model,
        loss=dc.models.losses.SparseSoftmaxCrossEntropy(),
        output_types=['prediction', 'loss'],
        batch_size=batch_size,
        **kwargs)


def DRAccuracy(y, y_pred):
  y = np.argmax(y, 1)
  y_pred = np.argmax(y_pred, 1)
  return accuracy_score(y, y_pred)


def DRSpecificity(y, y_pred):
  y_pred = (np.argmax(y_pred, 1) > 0) * 1
  y = (y > 0) * 1
  TN = sum((1 - y_pred) * (1 - y))
  N = sum(1 - y)
  return float(TN) / N


def DRSensitivity(y, y_pred):
  y = np.argmax(y, 1)
  y_pred = (np.argmax(y_pred, 1) > 0) * 1
  y = (y > 0) * 1
  TP = sum(y_pred * y)
  P = sum(y)
  return float(TP) / P


def ConfusionMatrix(y, y_pred):
  y = np.argmax(y, 1)
  y_pred = np.argmax(y_pred, 1)
  return confusion_matrix(y, y_pred)


def QuadWeightedKappa(y, y_pred):
  y = np.argmax(y, 1)
  y_pred = np.argmax(y_pred, 1)
  cm = confusion_matrix(y, y_pred)
  classes_y, counts_y = np.unique(y, return_counts=True)
  classes_y_pred, counts_y_pred = np.unique(y_pred, return_counts=True)
  E = np.zeros((classes_y.shape[0], classes_y.shape[0]))
  for i, c1 in enumerate(classes_y):
    for j, c2 in enumerate(classes_y_pred):
      E[c1, c2] = counts_y[i] * counts_y_pred[j]
  E = E / np.sum(E) * np.sum(cm)
  w = np.zeros((classes_y.shape[0], classes_y.shape[0]))
  for i in range(classes_y.shape[0]):
    for j in range(classes_y.shape[0]):
      w[i, j] = float((i - j)**2) / (classes_y.shape[0] - 1)**2
  re = 1 - np.sum(w * cm) / np.sum(w * E)
  return re


class DRAugment(layers.Layer):

  def __init__(self,
               augment,
               batch_size,
               distort_color=True,
               central_crop=True,
               size=(512, 512),
               **kwargs):
    """
    Parameters
    ----------
    augment: bool
      If to use data augmentation
    batch_size: int
      Number of images in the batch
    distort_color: bool
      If to apply random distortion on the color
    central_crop: bool
      If to randomly crop the sample around the center
    size: int
      Resolution of the input images(square)
    """
    self.augment = augment
    self.batch_size = batch_size
    self.distort_color = distort_color
    self.central_crop = central_crop
    self.size = size
    super(DRAugment, self).__init__(**kwargs)

  def call(self, inputs, training=True):
    parent_tensor = inputs / 255.0
    if not self.augment or not training:
      return parent_tensor
    else:

      def preprocess(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=np.random.randint(0, 4))
        if self.distort_color:
          img = tf.image.random_brightness(img, max_delta=32. / 255.)
          img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
          img = tf.clip_by_value(img, 0.0, 1.0)
        if self.central_crop:
          # sample cut ratio from a clipped gaussian
          img = tf.image.central_crop(img, np.clip(np.random.normal(1., 0.06), 0.8, 1.))
          img = tf.image.resize(tf.expand_dims(img, 0), tf.convert_to_tensor(self.size))[0]
        return img

      return tf.map_fn(preprocess, parent_tensor)

