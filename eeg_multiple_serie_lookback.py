from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from data_loading import csv_to_dataframe
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import mean_squared_error

from plotutils import plot_predictions

TRAIN_SPLIT = 0.67
LOOK_BACK = 1
INPUT_CSV_FILE = "data/sessions/subject_1.csv"
CSV_SEP = ","
INPUT_COLUMNS = [
    "Alfa1",
    "Alfa2",
    "Beta1",
    "Beta2",
    "Delta",
    "Gamma1",
    "Gamma2",
    "Theta"
]
PREDICTION_COLUMN = "Alfa2"


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataframe = pd.read_csv(INPUT_CSV_FILE, sep=CSV_SEP, na_values=-1)

m_dataset = np.zeros(shape=(len(dataframe[PREDICTION_COLUMN]), len(INPUT_COLUMNS)))
for i, column in enumerate(INPUT_COLUMNS):
    col_values = dataframe[column].values
    m_dataset[:, i] = np.array(col_values).transpose()
    m_dataset[:, i] = m_dataset[:, i].astype('float32')

dataset = dataframe[PREDICTION_COLUMN].values
dataset = np.array([dataset]).transpose()
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
m_dataset = scaler.fit_transform(m_dataset)

# split into train and test sets
train_size = int(len(dataset) * TRAIN_SPLIT)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

m_train = m_dataset[0:train_size, :]
m_test = m_dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
train_input, train_target = create_dataset(train, LOOK_BACK)
test_input, test_target = create_dataset(test, LOOK_BACK)

assert len(m_train) > 0, "Train array length must me > 0"

mt = m_train.shape[0]
n_features = m_train.shape[1]
m_train_input = np.zeros((mt - LOOK_BACK, LOOK_BACK, n_features))
m_train_target = np.zeros((mt - LOOK_BACK, n_features))
for i in range(n_features):
    train_col = m_train[:, i]
    train_col = np.array([train_col]).transpose()
    col_train_input, col_train_target = create_dataset(train_col, LOOK_BACK)
    m_train_input[:, :, i] = col_train_input
    m_train_target[:, i] = col_train_target

mv = m_test.shape[0]
m_test_input = np.zeros((mv - LOOK_BACK, LOOK_BACK, n_features))
m_test_target = np.zeros((mv - LOOK_BACK, n_features))
for i in range(n_features):
    test_col = m_test[:, i]
    test_col = np.array([test_col]).transpose()
    col_test_input, col_test_target = create_dataset(test_col, LOOK_BACK)
    m_test_input[:, :, i] = col_test_input
    m_test_target[:, i] = col_test_target

print(f"Train input:target shape = {train_input.shape}:{train_target.shape}")
print(f"Test input:target shape = {test_input.shape}:{test_target.shape}")
print(f"Train input:target shape = {m_train_input.shape}:{m_train_target.shape}")
print(f"Test input:target shape = {m_test_input.shape}:{m_test_target.shape}")

# reshape input to be [samples, time steps, features]
# Samples. One sequence is one sample. A batch is comprised of one or more samples.
# Time Steps. One time step is one point of observation in the sample.
# # Features. One feature is one observation at a time step.
# train_input = np.reshape(train_input, (train_input.shape[0], 1, n_features))
# test_input = np.reshape(test_input, (test_input.shape[0], 1, n_features))
#
# m_train_input = np.reshape(m_train_input, (m_train_input.shape[0], 1, m_train_input.shape[1]))
# m_test_input = np.reshape(m_test_input, (m_test_input.shape[0], 1, m_test_input.shape[1]))

print("Reshape input to be [samples, time steps, features]")
print(f"Train input:target shape after reshaping = {train_input.shape}:{train_target.shape}")
print(f"Test input:target shape after reshaping = {test_input.shape}:{test_target.shape}")

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(LOOK_BACK, n_features)))
model.add(Dense(n_features))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(m_train_input, m_train_target, epochs=100, batch_size=1, verbose=2)

# make predictions
m_trainPredict = model.predict(m_train_input)
m_testPredict = model.predict(m_test_input)
# invert predictions
m_trainPredict = scaler.inverse_transform(m_trainPredict)
m_train_target = scaler.inverse_transform(m_train_target)
m_testPredict = scaler.inverse_transform(m_testPredict)
m_test_target = scaler.inverse_transform(m_test_target)

train_score = np.zeros(n_features)
test_score = np.zeros(n_features)
for i in range(n_features):

    # calculate root mean squared error
    train_score[i] = math.sqrt(mean_squared_error(m_train_target[:, i], m_trainPredict[:, i]))
    print(f'Train Score {INPUT_COLUMNS[i]}: {train_score[i]} RMSE')
    test_score[i] = math.sqrt(mean_squared_error(m_test_target[:, i], m_testPredict[:, i]))
    print(f'Test Score {INPUT_COLUMNS[i]}: {test_score[i]} RMSE')

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(m_dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[LOOK_BACK:train_size, i] = m_trainPredict[:, i]
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(m_dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[train_size + LOOK_BACK:len(m_dataset), i] = m_testPredict[:, i]

    plt.title(f"Predictions {INPUT_COLUMNS[i]}")
    t_dataset = scaler.inverse_transform(m_dataset)
    plt.plot(t_dataset[:, i], label=f"{INPUT_COLUMNS[i]}", linestyle="-")
    plt.plot(trainPredictPlot[:, i], label="Train predictions", linestyle="-", fillstyle='none')
    plt.plot(testPredictPlot[:, i], label="Validation predictions", linestyle="-", fillstyle='none')

    plt.legend()
    plt.savefig(f'plots\\eeg_multiple_predictions_{INPUT_COLUMNS[i]}.png')
    plt.show()
    plt.close()

df = pd.DataFrame({"Train RMSE": train_score, "Validation RMSE": test_score}, index=INPUT_COLUMNS)
ax = df.plot.bar(color=["SkyBlue", "IndianRed"], rot=0, title="RMSE")
ax.set_xlabel("Feature")
plt.show()
