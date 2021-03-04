"""
Created on Mon Sep 10 06:12:11 2018

@author: zqwu
"""

import deepchem as dc
import numpy as np
import pandas as pd
import os
import logging
from model import DRModel, DRAccuracy, ConfusionMatrix, QuadWeightedKappa
from data import load_images_DR

RETRAIN = True

train, valid, test = load_images_DR(split='random', seed=123)
# Define and build model
model = DRModel(
    n_init_kernel=32,
    batch_size=32,
    learning_rate=1e-5,
    augment=True,
    model_dir='./test_model')
if not os.path.exists('./test_model'):
  os.mkdir('test_model')
if not RETRAIN:
  os.system("sh get_pretrained_model.sh")
  model.restore(checkpoint="./test_model/model-84384")
metrics = [
    dc.metrics.Metric(DRAccuracy, mode='classification'),
    dc.metrics.Metric(QuadWeightedKappa, mode='classification')
]
cm = [dc.metrics.Metric(ConfusionMatrix, mode='classification')]

logger = logging.getLogger('deepchem.models.tensorgraph.tensor_graph')
logger.setLevel(logging.DEBUG)
if RETRAIN:
  print("About to fit model for 10 epochs")
  model.fit(train, nb_epoch=10, checkpoint_interval=1000)
print("About to start train metrics evaluation")
print(model.evaluate(train, metrics, n_classes=5))
print("About to start valid metrics evaluation")
print(model.evaluate(valid, metrics, n_classes=5))
print("About to start valid confusion matrix evaluation")
print(model.evaluate(valid, cm, n_classes=5))
print("About to start test metrics evaluation")
print(model.evaluate(test, metrics, n_classes=5))
print("About to start test confusion matrix evaluation")
print(model.evaluate(test, cm, n_classes=5))
