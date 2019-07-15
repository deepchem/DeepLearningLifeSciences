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
model.build()
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
model.evaluate(train, metrics)
print("About to start valid metrics evaluation")
model.evaluate(valid, metrics)
print("About to start valid confusion matrix evaluation")
model.evaluate(valid, cm)
print("About to start test metrics evaluation")
model.evaluate(test, metrics)
print("About to start test confusion matrix evaluation")
model.evaluate(test, cm)
