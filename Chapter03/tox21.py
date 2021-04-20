# Train a model to predict toxicity.

import numpy as np
import deepchem as dc

# Load the data.

tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Build and train the model.

model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_sizes=[1000])
model.fit(train_dataset, nb_epoch=10)

# Evaluate it on the training and test sets.

metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print(train_scores)
print(test_scores)