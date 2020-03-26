# Train a model to predict transcription factor binding, based on both
# sequence and chromatin accessibility.

import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

# Build the model.

features = tf.keras.Input(shape=(101, 4))
accessibility = tf.keras.Input(shape=(1,))
prev = features
for i in range(3):
    prev = layers.Conv1D(filters=15, kernel_size=10, activation=tf.nn.relu, padding='same')(prev)
    prev = layers.Dropout(rate=0.5)(prev)
prev = layers.Concatenate()([layers.Flatten()(prev), accessibility])
logits = layers.Dense(units=1)(prev)
output = layers.Activation(tf.math.sigmoid)(logits)
keras_model = tf.keras.Model(inputs=[features, accessibility], outputs=[output, logits])
model = dc.models.KerasModel(
    keras_model,
    loss=dc.models.losses.SigmoidCrossEntropy(),
    output_types=['prediction', 'loss'],
    batch_size=1000,
    model_dir='chromatin')

# Load the data.

train = dc.data.DiskDataset('train_dataset')
valid = dc.data.DiskDataset('valid_dataset')
span_accessibility = {}
for line in open('accessibility.txt'):
    fields = line.split()
    span_accessibility[fields[0]] = float(fields[1])

# Define a generator function to produce batches.

def generate_batches(dataset, epochs):
    for epoch in range(epochs):
        for X, y, w, ids in dataset.iterbatches(batch_size=1000, pad_batches=True):
            yield ([X, np.array([span_accessibility[id] for id in ids])], [y], [w])

# Train the model, tracking its performance on the training and validation datasets.

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
for i in range(20):
    model.fit_generator(generate_batches(train, epochs=10))
    print(model.evaluate_generator(generate_batches(train, 1), [metric]))
    print(model.evaluate_generator(generate_batches(valid, 1), [metric]))
