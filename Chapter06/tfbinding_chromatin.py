# Train a model to predict transcription factor binding, based on both
# sequence and chromatin accessibility.

import deepchem as dc
import deepchem.models.tensorgraph.layers as layers
import tensorflow as tf
import numpy as np

# Build the model.

model = dc.models.TensorGraph(batch_size=1000, model_dir='chromatin')
features = layers.Feature(shape=(None, 101, 4))
accessibility = layers.Feature(shape=(None, 1))
labels = layers.Label(shape=(None, 1))
weights = layers.Weights(shape=(None, 1))
prev = features
for i in range(3):
    prev = layers.Conv1D(filters=15, kernel_size=10, activation=tf.nn.relu, padding='same', in_layers=prev)
    prev = layers.Dropout(dropout_prob=0.5, in_layers=prev)
prev = layers.Concat([layers.Flatten(prev), accessibility])
logits = layers.Dense(out_channels=1, in_layers=prev)
output = layers.Sigmoid(logits)
model.add_output(output)
loss = layers.SigmoidCrossEntropy(in_layers=[labels, logits])
weighted_loss = layers.WeightedError(in_layers=[loss, weights])
model.set_loss(weighted_loss)

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
            yield {
                features: X,
                accessibility: np.array([span_accessibility[id] for id in ids]),
                labels: y,
                weights: w
            }

# Train the model, tracking its performance on the training and validation datasets.

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
for i in range(20):
    model.fit_generator(generate_batches(train, epochs=10))
    print(model.evaluate_generator(generate_batches(train, 1), [metric], labels=[labels], weights=[weights]))
    print(model.evaluate_generator(generate_batches(valid, 1), [metric], labels=[labels], weights=[weights]))
