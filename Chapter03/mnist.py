# Train a model to recognize handwritten digits.

import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers

# Load the dataset and convert the labels to one-hot.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train = tf.one_hot(y_train, 10).numpy()
y_test = tf.one_hot(y_test, 10).numpy()
train_dataset = dc.data.NumpyDataset(x_train, y_train)
test_dataset = dc.data.NumpyDataset(x_test, y_test)

# Build the model.

features = tf.keras.Input(shape=(28, 28, 1))
conv2d_1 = layers.Conv2D(filters=32, kernel_size=5, activation=tf.nn.relu)(features)
conv2d_2 = layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu)(conv2d_1)
flatten = layers.Flatten()(conv2d_2)
dense1 = layers.Dense(units=1024, activation=tf.nn.relu)(flatten)
dense2 = layers.Dense(units=10, activation=None)(dense1)
output = layers.Activation(tf.math.softmax)(dense2)
keras_model = tf.keras.Model(inputs=features, outputs=[output, dense2])
model = dc.models.KerasModel(
    keras_model,
    loss=dc.models.losses.SoftmaxCrossEntropy(),
    output_types=['prediction', 'loss'],
    model_dir='mnist')

# Train the model.

model.fit(train_dataset, nb_epoch=10)

# Evaluate it on the training and test sets.

metric = dc.metrics.Metric(dc.metrics.accuracy_score)
train_scores = model.evaluate(train_dataset, [metric])
test_scores = model.evaluate(test_dataset, [metric])
print(train_scores)
print(test_scores)