# Importing core libraries
import numpy as np
from functions.dataInput import load2008,load2010, load2011, load2012, load2013, load2009, load2016, load2015, load2014, load2018, load2019, \
    temp2009, temp2011, temp2014, temp2015, temp2013, temp2016, temp2019
from sklearn.preprocessing import  StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPRegressor

import time
start_time = time.time()

loadTrainLagTarget = load2015['2015-01-06']
loadTrainLagMinus1 = load2015['2015-01-05']
loadTrainLagMinus7 = load2014['2014-12-31']
loadTrainLagMinus365 = load2014['2014-01-06']
tempTrainDayTarget = temp2015['2015-01-06']

loadTestLagTarget = load2019['2019-01-06']
loadTestLagMinus1 = load2019['2019-01-05']
loadTestLagMinus7 = load2018['2018-12-31']
loadTestLagMinus365 = load2018['2018-01-06']
tempTestDayTarget = temp2019['2019-01-06']

# Transforming to numpy arrays
loadTrainLagTarget = loadTrainLagTarget.to_numpy()
loadTrainLagMinus1 = loadTrainLagMinus1.to_numpy()
loadTrainLagMinus7 = loadTrainLagMinus7.to_numpy()
loadTrainLagMinus365 = loadTrainLagMinus365.to_numpy()
tempTrainDayTarget = tempTrainDayTarget.to_numpy()

loadTestLagTarget = loadTestLagTarget.to_numpy()
loadTestLagMinus1 = loadTestLagMinus1.to_numpy()
loadTestLagMinus7 = loadTestLagMinus7.to_numpy()
loadTestLagMinus365 = loadTestLagMinus365.to_numpy()
tempTestDayTarget = tempTestDayTarget.to_numpy()
#######################################################################################################################

# Creating Train Sets
data = []
data2 = []

for i in range(0, 24):
    temp1 = [loadTrainLagMinus1[i], loadTrainLagMinus7[i], loadTrainLagMinus365[i]]
    data.append(temp1)
data = np.reshape(data, (24, len(temp1)))

for i in range(0, 24):
    temp2 = loadTrainLagTarget[i]
    data2.append(temp2)
data2 = np.reshape(data2, (-1, 1))

trainX = data
trainY = data2
#######################################################################################################################

# Creating Test Sets
data = []
data2 = []

for i in range(0, 24):
    temp1 = [loadTestLagMinus1[i], loadTestLagMinus7[i], loadTestLagMinus365[i]]
    data.append(temp1)
data = np.reshape(data, (24, len(temp1)))

for i in range(0, 24):
    temp2 = loadTestLagTarget[i]
    data2.append(temp2)
data2 = np.reshape(data2, (-1, 1))
testX = data
testY = data2

#######################################################################################################################
# Scaling
scaler = StandardScaler()

trainX = scaler.fit_transform(trainX)
trainY = scaler.fit_transform(trainY)

testX = scaler.fit_transform(testX)
testY = scaler.fit_transform(testY)

#######################################################################################################################

# Creating a Wrapper to work around the layers problem

class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, layer1=100):
        self.layer1 = layer1




    def fit(self, X, y):
        model = MLPRegressor(
            hidden_layer_sizes=[self.layer1],
        max_iter=2000)
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


opt = BayesSearchCV(
    estimator=MLPWrapper(),
    search_spaces={
        'layer1': Integer(24, 256),




    },
    n_iter=128, verbose=2
)


opt.fit(trainX, trainY.ravel())
best_params = opt.best_params_
print(best_params)
print(opt.score(testX, testY))
print("--- %s seconds ---" % (time.time() - start_time))