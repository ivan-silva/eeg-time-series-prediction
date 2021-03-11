from sklearn.preprocessing import MinMaxScaler

from config.param import DATA_DIR
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math
from sklearn.metrics import mean_squared_error

from utils.plotutils import plot_predictions

TRAIN_SPLIT = 0.67
LOOK_BACK = 1
INPUT_CSV_FILE = f"{DATA_DIR}/sessions/subject_1.csv"
# INPUT_CSV_FILE = "{DATA_DIR}/airline-passengers.csv"
CSV_SEP = ","
PREDICTION_COLUMN = "Alfa2"


# PREDICTION_COLUMN = "Passengers"


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataframe = pd.read_csv(INPUT_CSV_FILE, sep=CSV_SEP, na_values=-1)
dataset = dataframe[PREDICTION_COLUMN].values
dataset = np.array([dataset]).T
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * TRAIN_SPLIT)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
train_input, train_target = create_dataset(train, LOOK_BACK)
test_input, test_target = create_dataset(test, LOOK_BACK)

print(f"Train input:target shape = {train_input.shape}:{train_target.shape}")
print(f"Test input:target shape = {test_input.shape}:{test_target.shape}")

# reshape input to be [samples, time steps, features]
train_input = np.reshape(train_input, (train_input.shape[0], 1, train_input.shape[1]))
test_input = np.reshape(test_input, (test_input.shape[0], 1, test_input.shape[1]))

print("Reshape input to be [samples, time steps, features]")
print(f"Train input:target shape after reshaping = {train_input.shape}:{train_target.shape}")
print(f"Test input:target shape after reshaping = {test_input.shape}:{test_target.shape}")

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, LOOK_BACK)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(train_input, train_target, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(train_input)
testPredict = model.predict(test_input)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
train_target = scaler.inverse_transform([train_target])
testPredict = scaler.inverse_transform(testPredict)
test_target = scaler.inverse_transform([test_target])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_target[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_target[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[LOOK_BACK:len(trainPredict) + LOOK_BACK, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (LOOK_BACK * 2) + 1:len(dataset) - 1, :] = testPredict

plot_predictions(scaler.inverse_transform(dataset), trainPredictPlot, testPredictPlot, PREDICTION_COLUMN)
