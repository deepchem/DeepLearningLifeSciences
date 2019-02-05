# Train a model to predict binding sites for the transcription factor JUND.

import deepchem as dc
import deepchem.models.tensorgraph.layers as layers
import tensorflow as tf

# Build the model.

model = dc.models.TensorGraph(batch_size=1000, model_dir='tf')
features = layers.Feature(shape=(None, 101, 4))
labels = layers.Label(shape=(None, 1))
weights = layers.Weights(shape=(None, 1))
prev = features
for i in range(3):
    prev = layers.Conv1D(filters=15, kernel_size=10, activation=tf.nn.relu, padding='same', in_layers=prev)
    prev = layers.Dropout(dropout_prob=0.5, in_layers=prev)
logits = layers.Dense(out_channels=1, in_layers=layers.Flatten(prev))
output = layers.Sigmoid(logits)
model.add_output(output)
loss = layers.SigmoidCrossEntropy(in_layers=[labels, logits])
weighted_loss = layers.WeightedError(in_layers=[loss, weights])
model.set_loss(weighted_loss)

# Load the data.

train = dc.data.DiskDataset('train_dataset')
valid = dc.data.DiskDataset('valid_dataset')

# Train the model, tracking its performance on the training and validation datasets.

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
for i in range(20):
    model.fit(train, nb_epoch=10)
    print(model.evaluate(train, [metric]))
    print(model.evaluate(valid, [metric]))
