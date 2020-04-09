import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import os
import re

RETRAIN = False

# Load the datasets.
image_dir = 'BBBC005_v1_images'
files = []
labels = []
for f in os.listdir(image_dir):
  if f.endswith('.TIF'):
    files.append(os.path.join(image_dir, f))
    labels.append(int(re.findall('_C(.*?)_', f)[0]))
loader = dc.data.ImageLoader()
dataset = loader.featurize(files, np.array(labels))
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, seed=123)

# Create the model.
features = tf.keras.Input(shape=(520, 696, 1))
prev_layer = features
for num_outputs in [16, 32, 64, 128, 256]:
  prev_layer = layers.Conv2D(num_outputs, kernel_size=5, strides=2, activation=tf.nn.relu)(prev_layer)
output = layers.Dense(1)(layers.Flatten()(prev_layer))
keras_model = tf.keras.Model(inputs=features, outputs=output)
learning_rate = dc.models.optimizers.ExponentialDecay(0.001, 0.9, 250)
model = dc.models.KerasModel(
    keras_model,
    loss=dc.models.losses.L2Loss(),
    learning_rate=learning_rate,
    model_dir='models/model')

if not os.path.exists('./models'):
  os.mkdir('models')
if not os.path.exists('./models/model'):
  os.mkdir('models/model')

if not RETRAIN:
  model.restore()

# Train it and evaluate performance on the test set.
if RETRAIN:
  print("About to fit model for 50 epochs")
  model.fit(train_dataset, nb_epoch=50)
y_pred = model.predict(test_dataset).flatten()
print(np.sqrt(np.mean((y_pred-test_dataset.y)**2)))
