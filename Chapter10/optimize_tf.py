# Identify inputs that maximize the output of the trained TF binding model.

import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

# Start by building the model.

features = tf.keras.Input(shape=(101, 4))
prev = features
for i in range(3):
    prev = layers.Conv1D(filters=15, kernel_size=10, activation=tf.nn.relu, padding='same')(prev)
    prev = layers.Dropout(rate=0.5)(prev)
logits = layers.Dense(units=1)(layers.Flatten()(prev))
output = layers.Activation(tf.math.sigmoid)(logits)
keras_model = tf.keras.Model(inputs=features, outputs=[output, logits])
model = dc.models.KerasModel(
    keras_model,
    loss=dc.models.losses.SigmoidCrossEntropy(),
    output_types=['prediction', 'loss'],
    batch_size=1000,
    model_dir='../Chapter06/tf')

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
