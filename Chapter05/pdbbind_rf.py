# Use a random forest to predict the PDBBind dataset.  First load the data.

import deepchem as dc
pdbbind_tasks, pdbbind_datasets, transformers = dc.molnet.load_pdbbind(featurizer="grid", split="random", subset="core")
train_dataset, valid_dataset, test_dataset = pdbbind_datasets

# Create and train the model.

from sklearn.ensemble import RandomForestRegressor
sklearn_model = RandomForestRegressor(n_estimators=100)
model = dc.models.SklearnModel(sklearn_model, model_dir="pdbbind_rf")
model.fit(train_dataset)

# Evaluate it.

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print("Train scores")
print(train_scores)
print("Test scores")
print(test_scores)
