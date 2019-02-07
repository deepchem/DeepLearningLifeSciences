import deepchem as dc
import deepchem.models.tensorgraph.layers as layers
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
learning_rate = dc.models.tensorgraph.optimizers.ExponentialDecay(0.001, 0.9, 250)
model = dc.models.TensorGraph(learning_rate=learning_rate, model_dir='models/model')
features = layers.Feature(shape=(None, 520, 696))
labels = layers.Label(shape=(None,))
prev_layer = features
for num_outputs in [16, 32, 64, 128, 256]:
  prev_layer = layers.Conv2D(num_outputs, kernel_size=5, stride=2, in_layers=prev_layer)
output = layers.Dense(1, in_layers=layers.Flatten(prev_layer))
model.add_output(output)
loss = layers.ReduceSum(layers.L2Loss(in_layers=(output, labels)))
model.set_loss(loss)

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
