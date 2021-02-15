from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from data_loading import csv_to_dataframe
import pandas
from zipfile import ZipFile
import os
import shutil
import pandas as pd
from keras import metrics
from matplotlib import pyplot as plt

from data_loading import csv_to_dataframe

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from plotutils import plot_confusion_matrix, plot_predictions

from sklearn.metrics import mean_squared_error


df = csv_to_dataframe("data\\sessions", "subject_4.csv")

# feature_keys = list(df.columns)
titles = feature_keys = ['Alfa1', 'Alfa2', 'Beta1', 'Beta2', 'Delta', 'Gamma1', 'Gamma2', 'Theta', 'Meditazione',
                         'Attenzione']
print(feature_keys)

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
index_key = "Session"

def show_raw_visualization(data):
    time_data = data[index_key]
    fig, axes = plt.subplots(
        nrows=5, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        print(t_data)
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title=f"{key}",
            rot=25,
        )
        ax.legend([feature_keys[i]])
    plt.tight_layout()
    plt.savefig('plots\\parameters_plot.png')
    plt.show()
    plt.close()

show_raw_visualization(df)

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=12, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=12)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.savefig('plots\\feature_correlation_heatmap.png', bbox_inches="tight")
    plt.show()
    plt.close()

show_heatmap(df)

split_fraction = .7
train_split = int(split_fraction * int(df.shape[0]))

# Sliding window configuration. Example:
# Example 1:
# Consider indices [0, 1, ... 99]. With sequence_length=10, sampling_rate=2, sequence_stride=3, shuffle=False.
# The dataset will yield batches of sequences composed of the following indices:
# First sequence:  [0  2  4  6  8 10 12 14 16 18]
# Second sequence: [3  5  7  9 11 13 15 17 19 21]
# Third sequence:  [6  8 10 12 14 16 18 20 22 24]
# ...
# Last sequence:   [78 80 82 84 86 88 90 92 94 96]
# In this case the last 3 data points are discarded since no full sequence can be generated to include them
# (the next sequence would have started at index 81, and thus its last step would have gone over 99).
sequence_length = 3
sampling_rate = 1
sequence_stride = 1

learning_rate = 0.001
batch_size = 1
epochs = 100

# Parameter filtering. We select only interesting parameters. In this case we are removing "Meditazione" and
# "Attenzione" because they are calculated.
print("\nParameter filtering =====================================================================================")
selected_parameters = [0, 1, 2, 3, 4, 5, 6, 7]
print(
    "The selected parameters are:",
    ", ".join([titles[i] for i in selected_parameters]),
)
selected_features = [feature_keys[i] for i in selected_parameters]
features = df[selected_features]
features.index = df[index_key]

# Data normalization
print("\nData normalization ======================================================================================")

# Standard score
def standard_score(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

print("Values before normalization", features)
df.plot(x='Session', title="Raw parameters")
plt.savefig('plots\\raw_parameters.png')
plt.show()
plt.close()

# features = standard_score(features.values, train_split)
scaler = MinMaxScaler((-1, 1))
features = scaler.fit_transform(features)

features = pd.DataFrame(features)
print("Values after normalization", features)
features.plot(title="Selected parameters, normalized")
plt.savefig('plots\\normalized_parameters.png')
plt.show()
plt.close()

# Train/validation split
all_data = features.loc[0:]
train_data = features.loc[0: train_split - 1]
val_data = features.loc[train_split:]
print(f"Total data {len(all_data)} elements, training data {len(train_data)} elements, "
      f"validation data {len(val_data)} elements")

# Training dataset

prediction_index = 7
x_train = train_data[[i for i in range(len(selected_parameters))]].values
y_train = features.iloc[0:train_split][[prediction_index]]

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=sampling_rate,
    sequence_stride=sequence_stride,
    batch_size=batch_size,
)
print('Train dataset', dataset_train)

# Validation dataset
x_end = len(val_data)

x_val = val_data.iloc[:x_end][[i for i in range(len(selected_parameters))]].values
y_val = features.iloc[train_split:][[prediction_index]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sequence_stride=sequence_stride,
    sampling_rate=sampling_rate,
    batch_size=batch_size,
)
print('Validation dataset', dataset_val)

for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "sessions_model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

def visualize_loss(loss_history):
    loss = loss_history.history["loss"]
    val_loss = loss_history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.title(f"Training loss and validation loss: {titles[prediction_index]}")
    plt.plot(epochs, loss, colors[0], label="Training loss")
    plt.plot(epochs, val_loss, colors[1], label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'plots\\loss_val_loss_{titles[prediction_index]}.png')
    plt.show()
    plt.close()

visualize_loss(history)

predictions = model.predict(dataset_val)
predictions_x = []
predictions_y = []
i = train_split
for x, y in dataset_val:
    single_batch_predictions = model.predict(x)
    print("Single batch prediction: ", single_batch_predictions)
    j = 0
    for prediction in single_batch_predictions:
        predictions_x.append(i)
        predictions_y.append(prediction)
        j = j+1
        i = i+1

print(f'There are {len(predictions)} predictions:', predictions)
plt.title(f"Predictions: {titles[prediction_index]}")
plt.plot(all_data[prediction_index], label="Complete dataset", linestyle="-", c=colors[6])
plt.plot(train_data[prediction_index], label="Train set", linestyle=":", marker='.', fillstyle='none')
plt.plot(val_data[prediction_index], label="Validation set", linestyle=":", marker='.', fillstyle='none',
         c=colors[5])
plt.plot(predictions_x, predictions_y, label=f"{titles[prediction_index]} predictions", linestyle="", marker='x', fillstyle='none')
plt.legend()
plt.savefig(f'plots\\dataset_predictions_{titles[prediction_index]}.png')
plt.show()
plt.close()
