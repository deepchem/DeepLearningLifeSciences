# Use a neural network to predict the PDBBind dataset.  First load the data.

import deepchem as dc
pdbbind_tasks, pdbbind_datasets, transformers = dc.molnet.load_pdbbind(featurizer="grid", split="random", subset="core")
train_dataset, valid_dataset, test_dataset = pdbbind_datasets

# Create and train the model.

n_features = train_dataset.X.shape[1]
model = dc.models.MultitaskRegressor(
        n_tasks=len(pdbbind_tasks),
        n_features=n_features,
        layer_sizes=[2000, 1000],
        dropouts=0.5,
        model_dir="pdbbind_nn",
        learning_rate=0.0003)
model.fit(train_dataset, nb_epoch=250)

# Evaluate it.

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print("Train scores")
print(train_scores)
print("Test scores")
print(test_scores)
