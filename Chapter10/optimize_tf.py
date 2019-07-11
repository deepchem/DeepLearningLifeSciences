# Identify inputs that maximize the output of the trained TF binding model.

import deepchem as dc
import deepchem.models.tensorgraph.layers as layers
import tensorflow as tf
import numpy as np

# Start by building the model.

model = dc.models.TensorGraph(batch_size=1000, model_dir='../Chapter06/tf')
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

# Reload the trained model parameters.  This assumes you already ran the tfbinding.py
# script in the Chapter06 directory to train the model.

model.restore()

# Start with a random sequence.

best_sequence = np.random.randint(4, size=101)
best_score = float(model.predict_on_batch([dc.metrics.to_one_hot(best_sequence, 4)]))

# Make random changes to it, and keep them if the output increases.

for step in range(1000):
    index = np.random.randint(101)
    base = np.random.randint(4)
    if best_sequence[index] != base:
      sequence = best_sequence.copy()
      sequence[index] = base
      score = float(model.predict_on_batch([dc.metrics.to_one_hot(sequence, 4)]))
      if score > best_score:
        best_sequence = sequence
        best_score = score
print('Best sequence:', ''.join(['ACGT'[i] for i in best_sequence]))
print('Best score:', score)
