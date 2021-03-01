import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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

# Test score prediction from eeg (Keras-Regression vs Multiple Regression)

# We must create our dataset from the csv. The objective is to generate a dataset like this

#  [
#  "Test score before"
#  "alpha1",
#  "alpha2",
#  "beta1",
#  "beta2",
#  "delta",
#  "gamma1",
#  "gamma2",
#  "theta"
#  ]
plot_prefix = "eeg_regression_"
input_csv_files = [
    "subject_1.csv",
    "subject_2.csv",
    "subject_3.csv",
    "subject_4.csv",
    "subject_5.csv",
    "subject_6.csv",
]

sel_features = [
    "Alfa1",
    "Alfa2",
    "Beta1",
    "Beta2",
    "Delta",
    "Gamma1",
    "Gamma2",
    "Theta"
]

target_labels = [
    "ICV",
    "IRP",
    "IML",
    "IVE",
    "QI",
]
csv_sep = ","
na_values = -1
data_folder = "data\\sessions\\"
n_features = len(sel_features)
n_files = len(input_csv_files)
n_targets = len(target_labels)
smoothening_factor = 5
print(f"Selected features:", sel_features)


# We reduce dataset to one value per parameter via a function
def flattening_function(params):
    return np.average(params)


print(f"Constructing dataset with {n_files} files, {n_features} features, "
      f"considering first {smoothening_factor} feature values.")

initial_targets = pd.read_csv('data\\sessions\\initial_targets.csv')
initial_targets.head()
targets = pd.read_csv('data\\sessions\\targets.csv')
targets.head()
for i, input_csv_file in enumerate(input_csv_files):
    # Load each file
    dataframe = pd.read_csv(f"{data_folder}{input_csv_file}", sep=csv_sep, na_values=na_values)

    # All dataset must be the same shape. We use the first dataset shape to initialize data structures and we save it
    # to check the subsequent datasets compliancy.
    if i == 0:
        m = dataframe.shape[0]
        first_features_mean = np.zeros(shape=(n_files, n_features))
        last_features_mean = np.zeros(shape=(n_files, n_features))
        n_smooth_values = m - (smoothening_factor - 1)
        smoothened_dataset = np.zeros(shape=(n_files, n_features, n_smooth_values))
    else:
        assert dataframe.shape[0] == m, f"All dataset must be the same length of {m}. The current dataset is " \
                                        f"{dataframe.shape[0]} lines long."

    # For each feature we flatten it in a single value
    for j, feature in enumerate(sel_features):
        col_values = dataframe[feature].values
        col_values = col_values.astype('float32')
        first_features_mean[i, j] = flattening_function(col_values[:smoothening_factor])
        last_features_mean[i, j] = flattening_function(col_values[(len(col_values) - smoothening_factor):])

        # We pick in sliding window style, averages from the dataset to seek a trend

        average_feature_values = np.zeros(shape=(n_smooth_values))
        for k in range(n_smooth_values):
            average_feature_values[k] = flattening_function(col_values[k:k + smoothening_factor])
        smoothened_dataset[i, j, :] = average_feature_values

print("Generated dataset: ", first_features_mean)
dataframe = pd.DataFrame(first_features_mean, columns=sel_features)
print(dataframe.describe())

ncols = 2
nrows = int(n_files / ncols)
figsize = (9 * ncols, 6 * nrows)
fig, axes = plt.subplots(
    nrows=nrows, figsize=figsize, ncols=ncols, dpi=160, facecolor="w", edgecolor="k"
)
fig.suptitle(f"Parametri ammorbiditi con fattore {smoothening_factor}")
for i in range(n_files):

    # Plot
    if n_files > 1:
        row = int(i // ncols)
        col = i % ncols
        cur_axes = axes[row, col]
    else:
        cur_axes = axes

    cur_axes.set_title(f"{input_csv_files[i]}")

    for j in range(n_features):
        cur_axes.plot(smoothened_dataset[i, j, :], label=f"{sel_features[j]}", linestyle="-")
    cur_axes.legend()

plt.savefig(f'plots\\{plot_prefix}_smoothed_dataset_{smoothening_factor}.png')
plt.show()
plt.close()

# Prendiamo le prime n e le ultime n sessioni ne facciamo la media e formiamo un dataset
start_features = list(map(lambda s: f"{s}_start", sel_features))
end_features = list(map(lambda s: f"{s}_end", sel_features))

s_e_dataframe = pd.DataFrame(
    data=np.hstack((first_features_mean, last_features_mean)),
    columns=np.hstack((start_features, end_features))
)
print(s_e_dataframe.head())

# Define model
model = Sequential()
model.add(Dense(500, input_dim=n_features * 2 + 1, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
# model.summary() #Print model Summary

# Effettuiamo il processo per ogni target
predictions = np.zeros((n_files, n_targets))
for i, target_label in enumerate(target_labels):

    print(f"===============================================================================")
    print(f"Predicting values for target parameter {target_label}")
    # Aggiungiamo il parametro iniziale al dataset
    p_dataframe = pd.DataFrame(
        data=np.hstack((s_e_dataframe.values, np.expand_dims(initial_targets[target_label], axis=1))),
        columns=np.hstack((s_e_dataframe.columns, target_label))
    )
    print("Complete dataset for feature")
    print(p_dataframe)

    # Abbiamo pochi dati. Cicliamo per vedere se otteniamo informazioni, dati n sample ne usiamo n-1 per il train
    # e 1 per il test.
    for j in range(n_files):
        print(f"-------------------------------------------------------------------------------")
        print(f"{target_label} dataset for subject {j+1}")

        X_train = p_dataframe.loc[p_dataframe.index != j].values
        y_train = targets[target_label].loc[targets.index != j].values
        y_train = np.expand_dims(y_train, axis=1)

        X_val = p_dataframe.loc[p_dataframe.index == j].values
        y_val = targets[target_label].loc[targets.index == j].values
        y_val = np.expand_dims(y_val, axis=1)

        # print(f"Train X shape: {X_train.shape}")
        # print(f"Train y:")
        # print(y_train)
        print(pd.DataFrame(X_train, columns=p_dataframe.columns))
        model.fit(X_train, y_train, epochs=20, verbose=0)

        prediction = model.predict(X_val)
        print(f"Predicted {target_label}={prediction}. Real value={y_val}. Error={y_val-prediction}")
        predictions[j, i] = prediction

predictions = pd.DataFrame(predictions, columns=target_labels)
print("Predictions")
print(predictions)
print("Real final values")
print(targets)

train_score = np.zeros(n_targets)
test_score = np.zeros(n_targets)
score_avg_increase = np.zeros(n_targets)
for i, target in enumerate(target_labels):
    test_score[i] = math.sqrt(mean_squared_error(targets[target], predictions[target]))
    score_avg_increase[i] = np.average(np.subtract(targets[target], initial_targets[target]))

print(test_score)
df = pd.DataFrame({"Train RMSE": train_score, "Validation RMSE": test_score, "Average increase": score_avg_increase}, index=target_labels)
ax = df.plot.bar(color=["SkyBlue", "IndianRed"], rot=0, title=f"Eeg regression RMSE")
ax.set_xlabel("Feature")
ax.set_xticklabels(target_labels, rotation=45)
plt.tight_layout()
plt.savefig(f'plots\\{plot_prefix}_RMSE.png', bbox_inches="tight")
plt.show()
