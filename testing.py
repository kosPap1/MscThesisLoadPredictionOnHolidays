# Importing core libraries
import numpy as np
from functions.dataInput import load2008,load2010, load2011, load2012, load2013, load2009, load2016, load2015, load2014, load2018, load2019, \
    temp2009, temp2011, temp2014, temp2015, temp2013, temp2016, temp2019
from sklearn.preprocessing import  StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.svm import LinearSVR, SVR
from sklearn.pipeline import Pipeline


import time
start_time = time.time()

loadTrainLagTarget = load2009['2009-03-02']
loadTrainLagMinus1 = load2009['2009-03-01']
loadTrainLagMinus7 = load2009['2009-02-23']
loadTrainLagMinus365 = load2008['2008-03-10']
tempTrainDayTarget = temp2009['2009-03-02']

loadTestLagTarget = load2019['2019-03-11']
loadTestLagMinus1 = load2019['2019-03-10']
loadTestLagMinus7 = load2019['2019-03-04']
loadTestLagMinus365 = load2018['2018-02-19']
tempTestDayTarget = temp2019['2019-03-11']

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

# pipeline class is used as estimator to enable
# search over different model types
pipe = Pipeline([
    ('model', SVR())
])

# single categorical value of 'model' parameter is
# sets the model class
# We will get ConvergenceWarnings because the problem is not well-conditioned.
# But that's fine, this is just an example.
linsvc_search = {
    'model': [LinearSVR(max_iter=1000)],
    'model__C': (1e-6, 1e+6, 'log-uniform'),
}

# explicit dimension classes can be specified like this
svc_search = {
    'model': Categorical([SVR()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1,8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}

opt = BayesSearchCV(
    pipe,
    # (parameter space, # of evaluations)
    [(svc_search, 40), (linsvc_search, 16)],
    cv=3, verbose=2
)

opt.fit(trainX, trainY.ravel())

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(testX, testY))
print("best params: %s" % str(opt.best_params_))