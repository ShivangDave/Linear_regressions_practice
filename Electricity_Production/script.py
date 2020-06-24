import numpy as np
import pandas as pd
import pdb

train_data = pd.read_csv('train_data.csv');
test_data = pd.read_csv('test_data.csv');

train_X = np.matrix(train_data)[:,range(0,train_data.shape[1]-1)]
train_y = np.matrix(train_data)[:,train_data.shape[1]-1]

test_X = np.matrix(test_data)

mu = train_X.mean(0)
sigma = train_X.std(0)

train_X_normalized = (train_X - mu) / sigma
unit_X = np.ones((train_X_normalized.shape[0],1))
train_X_normalized = np.hstack((unit_X,train_X_normalized))

test_X_normalized = (test_X - mu) / sigma
test_unit_X = np.ones((test_X_normalized.shape[0],1))
test_X_normalized = np.hstack((test_unit_X,test_X_normalized))

m = train_X_normalized.shape[0]
theta = np.zeros((train_X_normalized.shape[1],1))

predictions = train_X_normalized * theta
squareError = np.square(predictions - train_y)
J = 1/(2*m) * sum(squareError)

xTransposeX = train_X_normalized.transpose() * train_X_normalized
inverseXTransposeX = np.linalg.inv(xTransposeX)
xTransposeY = train_X_normalized.transpose() * train_y

ne_theta = inverseXTransposeX * xTransposeY

predictions_with_ne = test_X_normalized * ne_theta

pdb.set_trace()
