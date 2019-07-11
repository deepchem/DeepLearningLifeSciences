# Estimate the uncertainty in a model's predictions.

import deepchem as dc
import numpy as np

# Start by creating and training the model.

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets
model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2, uncertainty=True)
model.fit(train_dataset, nb_epoch=100)

# Predict values and uncertainties on the test set.

y_pred, y_std = model.predict_uncertainty(test_dataset)

# Plot a graph of absolute error versus predicted uncertainty.

import matplotlib.pyplot as plot
plot.scatter(y_std, np.abs(y_pred-test_dataset.y))
plot.plot([0, 0.7], [0, 1.4], 'k:')
plot.xlim([0.1, 0.7])
plot.xlabel('Predicted Standard Deviation')
plot.ylabel('Absolute Error')
plot.show()
