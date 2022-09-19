import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from sklearn import svm
from functions.dataInput import  load2012, load2013, load2018, load2019, temp2016, temp2019
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Lags creation
loadTrainLagTarget = load2013['2013-12-25']
loadTrainLagMinus1 = load2013['2013-12-24']
loadTrainLagMinus7 = load2013['2013-12-18']
loadTrainLagMinus365 = load2012['2012-12-25']
tempTrainDayTarget = temp2016['2016-12-25']

loadTestLagTarget = load2019['2019-12-25']
loadTestLagMinus1 = load2019['2019-12-24']
loadTestLagMinus7 = load2019['2019-12-18']
loadTestLagMinus365 = load2018['2018-12-25']
tempTestDayTarget = temp2019['2019-12-25']

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
model = xgb.XGBRegressor(n_estimators=292, learning_rate=0.900438, max_depth=3,  booster='dart',
                         reg_alpha= 0.01, reg_lambda = 15.5664, base_score=5)

model.fit(trainX, trainY)


# Predicting Next Days load
y_pred = model.predict(testX)

#######################################################################################################################
MLPreg = MLPRegressor(hidden_layer_sizes=(233, 512, 512), random_state=0, early_stopping=True,
                      solver='adam').fit(trainX, trainY.ravel())

y_pred_MLP = MLPreg.predict(testX)

#######################################################################################################################
# running the SVM predictor

modelSVM = svm.SVR(kernel='rbf', C=33.2408, gamma='auto')
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

print('MAPE for Crhistmas with XGBoost is ' + str(mean_absolute_percentage_error(y_pred, testY) * 100))
print('MAPE for Crhistmas with MLP is ' + str(mean_absolute_percentage_error(y_pred_MLP, testY) * 100))
print('MAPE for Crhistmas with SVM is ' + str(mean_absolute_percentage_error(y_predSVM, testY) * 100))

plt.plot(testY, label='actual value')
plt.plot(y_pred, label='prediction')
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









