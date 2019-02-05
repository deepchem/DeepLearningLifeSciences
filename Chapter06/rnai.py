# Train a model to predict how well sequences will work for RNA interference.

import deepchem as dc
import deepchem.models.tensorgraph.layers as layers
import tensorflow as tf
import matplotlib.pyplot as plot

# Build the model.

model = dc.models.TensorGraph(model_dir='rnai')
features = layers.Feature(shape=(None, 21, 4))
labels = layers.Label(shape=(None, 1))
prev = features
for i in range(2):
    prev = layers.Conv1D(filters=10, kernel_size=10, activation=tf.nn.relu, padding='same', in_layers=prev)
    prev = layers.Dropout(dropout_prob=0.3, in_layers=prev)
output = layers.Dense(out_channels=1, activation_fn=tf.sigmoid, in_layers=layers.Flatten(prev))
model.add_output(output)
loss = layers.ReduceMean(layers.L2Loss(in_layers=[labels, output]))
model.set_loss(loss)

# Load the data.

train = dc.data.DiskDataset('train_siRNA')
valid = dc.data.DiskDataset('valid_siRNA')

# Train the model, tracking its performance on the training and validation datasets.

metric = dc.metrics.Metric(dc.metrics.pearsonr, mode='regression')
for i in range(20):
    model.fit(train, nb_epoch=10)
    print(model.evaluate(train, [metric])['pearsonr'][0])
    print(model.evaluate(valid, [metric])['pearsonr'][0])
