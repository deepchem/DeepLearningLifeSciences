import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import os
import re

RETRAIN = False

# Load the datasets.
image_dir = 'BBBC005_v1_images'
label_dir = 'BBBC005_v1_ground_truth'
rows = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P')
blurs = (1, 4, 7, 10, 14, 17, 20, 23, 26, 29, 32, 35, 39, 42, 45, 48)
files = []
labels = []
for f in os.listdir(label_dir):
  if f.endswith('.TIF'):
    for row, blur in zip(rows, blurs):
      fname = f.replace('_F1', '_F%d'%blur).replace('_A', '_%s'%row)
      files.append(os.path.join(image_dir, fname))
      labels.append(os.path.join(label_dir, f))
dataset = dc.data.ImageDataset(files, labels)
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, seed=123)

# Create the model.
features = tf.keras.Input(shape=(520, 696, 1))
# Downsample three times.
conv1 = layers.Conv2D(16, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(features/255.0)
conv2 = layers.Conv2D(32, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(conv1)
conv3 = layers.Conv2D(64, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(conv2)
# Do a 1x1 convolution.
conv4 = layers.Conv2D(64, kernel_size=1, strides=1)(conv3)
# Upsample three times.
concat1 = layers.Concatenate(axis=3)([conv3, conv4])
deconv1 = layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(concat1)
concat2 = layers.Concatenate(axis=3)([conv2, deconv1])
deconv2 = layers.Conv2DTranspose(16, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(concat2)
concat3 = layers.Concatenate(axis=3)([conv1, deconv2])
deconv3 = layers.Conv2DTranspose(1, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(concat3)
# Compute the final output.
concat4 = layers.Concatenate(axis=3)([features, deconv3])
logits = layers.Conv2D(1, kernel_size=5, strides=1, padding='same')(concat4)
output = layers.Activation(tf.math.sigmoid)(logits)
keras_model = tf.keras.Model(inputs=features, outputs=[output, logits])
learning_rate = dc.models.optimizers.ExponentialDecay(0.01, 0.9, 250)
model = dc.models.KerasModel(
    keras_model,
    loss=dc.models.losses.SigmoidCrossEntropy(),
    output_types=['prediction', 'loss'],
    learning_rate=learning_rate,
    model_dir='models/segmentation')

if not os.path.exists('./models'):
  os.mkdir('models')
if not os.path.exists('./models/segmentation'):
  os.mkdir('models/segmentation')

if not RETRAIN:
  model.restore()

# Train it and evaluate performance on the test set.
if RETRAIN:
  print("About to fit model for 50 epochs")
  model.fit(train_dataset, nb_epoch=50, checkpoint_interval=100)
scores = []
for x, y, w, id in test_dataset.itersamples():
  y_pred = model.predict_on_batch([x]).squeeze()
  scores.append(np.mean((y>0) == (y_pred>0.5)))
print(np.mean(scores))

