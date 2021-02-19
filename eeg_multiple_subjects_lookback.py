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

VERBOSE = 2

INPUT_CSV_FILE = "data/sessions/multi_subject_alfa_1.csv"
CSV_SEP = ","
sel_features = [
    "Subject1, Alfa1",
    "Subject2, Alfa1",
    "Subject3, Alfa1",
    "Subject4, Alfa1",
    "Subject5, Alfa1",
    "Subject6, Alfa1",
]

TRAIN_SPLIT = 0.67
LOOK_BACK = 1
BATCH_SIZE = 1
EPOCHS = 100
PLOT_PREFIX = "eeg_multiple_subjects"

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
m = dataframe.shape[0]
n_features = len(sel_features)
dataset = np.zeros(shape=(m, n_features))

print(f"Only {n_features} of {len(dataframe.index)} total feature will be used.")
print(f"Selected features:", sel_features)

# Choose only selected features in dataset
for i, feature in enumerate(sel_features):
    col_values = dataframe[feature].values
    dataset[:, i] = np.array(col_values).transpose()
    dataset[:, i] = dataset[:, i].astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(m * TRAIN_SPLIT)
train = dataset[0:train_size, :]
test = dataset[train_size:m, :]

# reshape into X=t and Y=t+1
assert len(train) > 0, "Train array length must me > 0"

mt = train_size
train_input = np.zeros((mt - LOOK_BACK, LOOK_BACK, n_features))
train_target = np.zeros((mt - LOOK_BACK, n_features))
for i in range(n_features):
    train_col = train[:, i]
    train_col = np.array([train_col]).transpose()
    col_train_input, col_train_target = create_dataset(train_col, LOOK_BACK)
    train_input[:, :, i] = col_train_input
    train_target[:, i] = col_train_target

mv = test.shape[0]
test_input = np.zeros((mv - LOOK_BACK, LOOK_BACK, n_features))
test_target = np.zeros((mv - LOOK_BACK, n_features))
for i in range(n_features):
    test_col = test[:, i]
    test_col = np.array([test_col]).transpose()
    col_test_input, col_test_target = create_dataset(test_col, LOOK_BACK)
    test_input[:, :, i] = col_test_input
    test_target[:, i] = col_test_target

print(f"Train input:target shape = {train_input.shape}:{train_target.shape}")
print(f"Test input:target shape = {test_input.shape}:{test_target.shape}")

# Input has to be in form [samples, time steps, sel_features]
#   - Samples. One sequence is one sample. A batch is comprised of one or more samples.
#   - Time Steps. One time step is one point of observation in the sample.
#   - Features. One feature is one observation at a time step.

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(LOOK_BACK, n_features)))
model.add(Dense(n_features))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(train_input, train_target, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

# make predictions
train_predict = model.predict(train_input)
test_predict = model.predict(test_input)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
train_target = scaler.inverse_transform(train_target)
test_predict = scaler.inverse_transform(test_predict)
test_target = scaler.inverse_transform(test_target)

train_score = np.zeros(n_features)
test_score = np.zeros(n_features)

ncols = 2
nrows = int(n_features / ncols)
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
)
fig.suptitle(f'Predictions, lookback={LOOK_BACK}, {EPOCHS} epochs')
for i in range(n_features):
    # calculate root mean squared error
    train_score[i] = math.sqrt(mean_squared_error(train_target[:, i], train_predict[:, i]))
    print(f'Train Score {sel_features[i]}: {train_score[i]} RMSE')
    test_score[i] = math.sqrt(mean_squared_error(test_target[:, i], test_predict[:, i]))
    print(f'Test Score {sel_features[i]}: {test_score[i]} RMSE')

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[LOOK_BACK:train_size, i] = train_predict[:, i]
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[train_size + LOOK_BACK:len(dataset), i] = test_predict[:, i]
    t_dataset = scaler.inverse_transform(dataset)

    # Plot
    row = int(i // ncols)
    col = i % ncols
    axes[row, col].set_title(f"{sel_features[i]}")
    axes[row, col].plot(t_dataset[:, i], label=f"{sel_features[i]}", linestyle="-")
    axes[row, col].plot(trainPredictPlot[:, i], label="Train predictions", linestyle="-", fillstyle='none')
    axes[row, col].plot(testPredictPlot[:, i], label="Validation predictions", linestyle="-", fillstyle='none')
    axes[row, col].legend()

plt.savefig(f'plots\\{PLOT_PREFIX}_predictions_{LOOK_BACK}_{EPOCHS}.png')
plt.show()
plt.close()

df = pd.DataFrame({"Train RMSE": train_score, "Validation RMSE": test_score}, index=sel_features)
ax = df.plot.bar(color=["SkyBlue", "IndianRed"], rot=0, title="RMSE")
ax.set_xlabel("Feature")
ax.set_xticklabels(sel_features, rotation=45)
plt.tight_layout()
plt.savefig(f'plots\\{PLOT_PREFIX}_RMSE_{LOOK_BACK}_{EPOCHS}.png', bbox_inches="tight")
plt.show()
