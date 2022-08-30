import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from sklearn import svm
from functions.dataInput import load2008,  load2009, load2018, load2019, temp2009, temp2019
from sklearn.preprocessing import  StandardScaler
from sklearn.neural_network import MLPRegressor

# Lags creation
loadTrainLagTarget = load2009['2009-10-28']
loadTrainLagMinus1 = load2009['2009-10-27']
loadTrainLagMinus7 = load2009['2009-10-21']
loadTrainLagMinus365 = load2008['2008-10-28']
tempTrainDayTarget = temp2009['2009-10-28']

loadTestLagTarget = load2019['2019-10-28']
loadTestLagMinus1 = load2019['2019-10-27']
loadTestLagMinus7 = load2019['2019-10-21']
loadTestLagMinus365 = load2018['2018-10-28']
tempTestDayTarget = temp2019['2019-10-28']

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
    temp3 = [loadTestLagMinus1[i], loadTestLagMinus7[i], loadTestLagMinus365[i]]
    data.append(temp3)
data = np.reshape(data, (24, len(temp1)))

for i in range(0, 24):
    temp4 = loadTestLagTarget[i]
    data2.append(temp4)
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

# Model training
model = xgb.XGBRegressor(base_score = 1.053902497381037,
             colsample_bytree = 0.6015271536191487,
             learning_rate = 1.28,
             max_depth = 1,
             n_estimators = 124,
             reg_alpha = 0.2110510434486649,
             reg_lambda= 9.848736385323008,
             subsample = 0.42872753557805143, booster='gbtree')
model.fit(trainX, trainY)

# Predicting Next Days load
y_pred = model.predict(testX)

#######################################################################################################################
# running the MLP regression
MLPreg = MLPRegressor(hidden_layer_sizes=(12, 213, 156), random_state=0, ).fit(trainX, trainY.ravel())

y_pred_MLP = MLPreg.predict(testX)
#######################################################################################################################
# running the SVM predictor

modelSVM = svm.SVR(C= 53.196, degree=4)
modelSVM.fit(trainX, trainY.ravel())
y_predSVM = modelSVM.predict(testX)


# Inverse transform
y_pred = np.reshape(y_pred, (-1, 1))
y_pred_MLP = np.reshape(y_pred_MLP, (-1, 1))
y_predSVM = np.reshape(y_predSVM, (-1, 1))

y_pred = scaler.inverse_transform(y_pred)
y_pred_MLP = scaler.inverse_transform(y_pred_MLP)
y_predSVM = scaler.inverse_transform(y_predSVM)
testY = scaler.inverse_transform(testY)

print('MAPE for Oxi Day with XGBoost is ' + str(mean_absolute_percentage_error(y_pred, testY) * 100))
print('MAPE for Oxi Day with MLP is ' + str(mean_absolute_percentage_error(y_pred_MLP, testY) * 100))
print('MAPE for Oxi Day with SVM is ' + str(mean_absolute_percentage_error(y_predSVM, testY) * 100))

plt.plot(testY, label='actual value')
plt.plot(y_pred, label='XGB')
plt.legend()
plt.show()

plt.plot(y_pred_MLP, label = 'MLP')
plt.plot(testY, label = 'actual')
plt.legend()
plt.show()

plt.plot(y_predSVM, label = 'SVM')
plt.plot(testY, label = 'actual')
plt.legend()
plt.show()







