import os

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from data_loading import csv_to_dataframe
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import mean_squared_error

from plotutils import plot_predictions


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def generic_multiple_series_lookback(
        input_csv_file,
        sel_features,
        csv_sep=",",
        train_split=0.67,
        look_back=1,
        lstm_units=4,
        batch_size=1,
        epochs=100,
        verbose=2,
        plot_prefix="",
        na_values=-1

):
    # Cuda setup
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # fix random seed for reproducibility
    np.random.seed(7)

    # load the dataset
    dataframe = pd.read_csv(input_csv_file, sep=csv_sep, na_values=na_values)
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
    train_size = int(m * train_split)
    train = dataset[0:train_size, :]
    test = dataset[train_size:m, :]

    # reshape into X=t and Y=t+1
    assert len(train) > 0, "Train array length must me > 0"

    mt = train_size
    train_input = np.zeros((mt - look_back, look_back, n_features))
    train_target = np.zeros((mt - look_back, n_features))
    for i in range(n_features):
        train_col = train[:, i]
        train_col = np.array([train_col]).transpose()
        col_train_input, col_train_target = create_dataset(train_col, look_back)
        train_input[:, :, i] = col_train_input
        train_target[:, i] = col_train_target

    mv = test.shape[0]
    test_input = np.zeros((mv - look_back, look_back, n_features))
    test_target = np.zeros((mv - look_back, n_features))
    for i in range(n_features):
        test_col = test[:, i]
        test_col = np.array([test_col]).transpose()
        col_test_input, col_test_target = create_dataset(test_col, look_back)
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
    model.add(LSTM(lstm_units, input_shape=(look_back, n_features)))
    model.add(Dense(n_features))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    
    model_path = f'models/{plot_prefix}_model'
    if os.path.isdir(model_path):
        print('Model checkpoint found, skipping training')
        model = keras.models.load_model(model_path)
    else:
        print('Model checkpoint not found, training model...')
        model.fit(train_input, train_target, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(model_path)

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

    nrows = int(n_features / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(15, 20), dpi=300, facecolor="w", edgecolor="k"
    )
    fig.suptitle(f'Predictions, lookback={look_back}, {epochs} epochs')
    for i in range(n_features):
        # calculate root mean squared error
        train_score[i] = math.sqrt(mean_squared_error(train_target[:, i], train_predict[:, i]))
        print(f'Train Score {sel_features[i]}: {train_score[i]} RMSE')
        test_score[i] = math.sqrt(mean_squared_error(test_target[:, i], test_predict[:, i]))
        print(f'Test Score {sel_features[i]}: {test_score[i]} RMSE')

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:train_size, i] = train_predict[:, i]
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[train_size + look_back:len(dataset), i] = test_predict[:, i]
        t_dataset = scaler.inverse_transform(dataset)

        # Plot
        row = int(i // ncols)
        col = i % ncols
        axes[row, col].set_title(f"{sel_features[i]}")
        axes[row, col].plot(t_dataset[:, i], label=f"{sel_features[i]}", linestyle="-")
        axes[row, col].plot(trainPredictPlot[:, i], label="Train predictions", linestyle="-", fillstyle='none')
        axes[row, col].plot(testPredictPlot[:, i], label="Validation predictions", linestyle="-", fillstyle='none')
        axes[row, col].legend()

    plt.savefig(f'plots\\{plot_prefix}_predictions_{look_back}_{epochs}.png')
    plt.show()
    plt.close()

    df = pd.DataFrame({"Train RMSE": train_score, "Validation RMSE": test_score}, index=sel_features)
    ax = df.plot.bar(color=["SkyBlue", "IndianRed"], rot=0, title=f"RMSE, lookback={look_back}, {epochs} epochs")
    ax.set_xlabel("Feature")
    ax.set_xticklabels(sel_features, rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots\\{plot_prefix}_RMSE_{look_back}_{epochs}.png', bbox_inches="tight")
    plt.show()
