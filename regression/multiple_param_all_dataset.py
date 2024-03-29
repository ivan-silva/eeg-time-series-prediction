from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from config.param import PLOT_DIR, DATA_DIR
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import math
from sklearn.metrics import mean_squared_error

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
data_dir = f'{DATA_DIR}/sessions/'
plot_dir = f'{PLOT_DIR}/'
n_features = len(sel_features)
n_subjects = len(input_csv_files)
n_targets = len(target_labels)

print(f"Selected features:", sel_features)
# Run configuration
smoothing_factor = 77
epochs = 1000
verbose = 2


# We reduce dataset to one value per parameter via a function
def smoothing_function(params):
    return np.average(params)


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

# Set targets and initial targets
initial_targets = pd.read_csv(f'{data_dir}initial_targets.csv')
initial_targets.head()
targets = pd.read_csv(f'{data_dir}targets.csv')
targets.head()

# initial_targets = scaler.fit_transform(initial_targets.values)
# targets = scaler.fit_transform(targets.values)

# All files must be the same shape. We use the first dataset shape to initialize data structures and we save it
# to check the subsequent datasets compliancy.
dataframe = pd.read_csv(f"{data_dir}{input_csv_files[0]}", sep=csv_sep, na_values=na_values)
m = dataframe.shape[0]

# Dataset generation (without targets)
first_features_mean = np.zeros(shape=(n_subjects, n_features))
last_features_mean = np.zeros(shape=(n_subjects, n_features))
splits = m
split_features = np.zeros((n_subjects, n_features * splits))
split_features_names = []
for i, input_csv_file in enumerate(input_csv_files):
    # Load each file
    dataframe = pd.read_csv(f"{data_dir}{input_csv_file}", sep=csv_sep, na_values=na_values)

    assert dataframe.shape[0] == m, f"All dataset must be the same length of {m}. The current dataset is " \
                                    f"{dataframe.shape[0]} lines long."

    for split in range(splits):

        current_split = m // splits * split

        for j, feature in enumerate(sel_features):
            col_values = dataframe[feature].values
            col_values = col_values.astype('float32')
            # first_features_mean[i, j] = smoothing_function(col_values[:smoothing_factor])
            # last_features_mean[i, j] = smoothing_function(col_values[(len(col_values) - smoothing_factor):])
            split_features[i, (split*n_features)+j] = col_values[split]
            if i == 0:
                split_features_names.append(f"{feature}_{split}")

# split_features_names = list(map(lambda feature_name: f"{feature_name}_start", sel_features))
# split_feature_names = list(map(lambda feature_name: f"{feature_name}_end", sel_features))
s_e_dataframe = pd.DataFrame(
    data=split_features,
    columns=split_features_names
)
print(s_e_dataframe.head())


# Output dataset
predictions = np.zeros((n_subjects, n_targets))

errors_shape = (n_subjects, n_targets)
test_rmse = np.zeros(errors_shape)
test_mae = np.zeros(errors_shape)
test_rmse_e = np.zeros(errors_shape)
test_mae_e = np.zeros(errors_shape)

# Effettuiamo il processo per ogni target
for i, target_label in enumerate(target_labels):

    print(f"===============================================================================")
    print(f"Predicting values for target parameter {target_label}")
    # Complete train dataset generation (with initial targets)
    p_dataframe = pd.DataFrame(
        data=np.hstack((s_e_dataframe.values, np.expand_dims(initial_targets[target_label], axis=1))),
        columns=np.hstack((s_e_dataframe.columns, target_label))
    )
    print("Complete dataset for feature")
    print(p_dataframe)

    # Specific train set for n-1 subjects with 1 subject as validation
    for j in range(n_subjects):
        print(f"-------------------------------------------------------------------------------")
        print(f"{target_label} dataset for subject {j + 1}")

        # Train set generation with index != j
        X_train = p_dataframe.loc[p_dataframe.index != j].values
        y_train = targets[target_label].loc[targets.index != j].values
        y_train = np.expand_dims(y_train, axis=1)

        # Validation set generation with index == j
        X_val = p_dataframe.loc[p_dataframe.index == j].values
        y_val = targets[target_label].loc[targets.index == j].values
        y_val = np.expand_dims(y_val, axis=1)

        print(f"Train X: {X_train.shape}")
        print(pd.DataFrame(X_train, columns=p_dataframe.columns))
        print(f"Train y:")
        print(y_train)

        # Train
        # print(f"Replay = {r}")
        # Model definition
        model = Sequential()
        model.add(Dense(500, input_dim=n_features * splits + 1, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
        # model.summary()
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=verbose)

        # Predict
        prediction = model.predict(X_val)
        print(f"Predicted {target_label}={prediction}. Real value={y_val}. Error={y_val - prediction}")
        predictions[j, i] = prediction

        # Train errors
        train_score = np.zeros(n_targets)
        # TODO

        # # Validation errors
        # for t, target in enumerate(target_labels):
        #     test_rmse_i[r, t] = math.sqrt(mean_squared_error(targets[target], prediction))
        #     test_mae_i[r, t] = mean_absolute_error(targets[target], predictions[target])

predictions = pd.DataFrame(predictions, columns=target_labels)
print("Predictions")
print(predictions)
print("Real final values")
print(targets)


# Trend prediction
trend_real = np.array(initial_targets) < np.array(targets.values)
trend_predicted = np.array(initial_targets) < np.array(predictions.values)
trend_correct = trend_real == trend_predicted
trend_percentage = 100/(trend_real.shape[0] * trend_real.shape[1]) * np.sum(trend_correct)
print(f"Trend prediction {trend_percentage}% correct")
print(trend_correct)

# Target errors
rmse = np.zeros(n_targets)
mae = np.zeros(n_targets)
for i, target in enumerate(target_labels):
    rmse[i] = math.sqrt(mean_squared_error(targets[target], predictions[target]))
    mae[i] = mean_absolute_error(targets[target], predictions[target])
print("Target RMSE")
print(rmse)
print(" Target MAE")
print(mae)

df = pd.DataFrame({
    "Validation RMSE": rmse,
    "Validation MAE": mae,
}, index=target_labels)
ax = df.plot.bar(color=["IndianRed", "Brown", "SkyBlue"], rot=0, title=f"Target errors, {epochs} epochs")
ax.set_xlabel("Feature")
ax.set_xticklabels(target_labels, rotation=45)
plt.tight_layout()
plt.savefig(f'{plot_dir}{plot_prefix}_errors_{epochs}.png', bbox_inches="tight")
plt.show()

# Subject errors
train_rmse_s = np.zeros(n_subjects)
train_mae_s = np.zeros(n_subjects)
test_rmse_s = np.zeros(n_subjects)
test_mae_s = np.zeros(n_subjects)
for i in range(n_subjects):
    train_rmse_s[i] = math.sqrt(mean_squared_error(targets.iloc[i].values, predictions.iloc[i].values))
    train_mae_s[i] = mean_absolute_error(targets.iloc[i].values, predictions.iloc[i].values)
    test_rmse_s[i] = math.sqrt(mean_squared_error(targets.iloc[i].values, predictions.iloc[i].values))
    test_mae_s[i] = mean_absolute_error(targets.iloc[i].values, predictions.iloc[i].values)
print("Subject RMSE")
print(test_rmse_s)
print("Subject MAE")
print(test_mae_s)

df = pd.DataFrame({
    "Train RMSE": train_rmse_s,
    "Train MAE": train_rmse_s,
    "Validation RMSE": test_rmse_s,
    "Validation MAE": test_mae_s,
})
ax = df.plot.bar(color=["SkyBlue", "IndianRed", "Brown"], rot=0, title=f"Subject errors, {epochs} epochs")
ax.set_xlabel("Feature")
plt.tight_layout()
plt.savefig(f'{plot_dir}{plot_prefix}_subject_errors_{epochs}.png', bbox_inches="tight")
plt.show()

#
#     test_rmse[:, i] = np.average(test_rmse_i, axis=0)
#     test_rmse_e[:, i] = stats.sem(test_rmse_i, axis=0)
#     test_mae[:, i] = np.average(test_mae_i, axis=0)
#     test_mae_e[:, i] = stats.sem(test_mae_i, axis=0)
#
# print("Test RMSE")
# print(test_rmse)
# print("Test MAE")
# print(test_mae)
#
# # Average increase, for reference
# for i, target in enumerate(target_labels):
#     score_avg_increase = np.zeros(n_targets)
#     score_avg_increase[i] = np.average(np.subtract(targets[target], initial_targets[target]))
# for i, target_label in enumerate(target_labels):
#     pl_i = plt.errorbar(range(test_rmse.shape[1]), test_rmse[i, :], test_rmse_e[i, :], capsize=5, capthick=1,
#                         label=target_label)
#     # plt.plot(test_rmse[i, :], label=f"Test RMSE {target_label}", linestyle="-", fillstyle='none')
#     plt.legend()
#     plt.show()
#
# for i, target_label in enumerate(target_labels):
#     pl_i = plt.errorbar(range(test_rmse.shape[1]), test_rmse[i, :], test_rmse_e[i, :], capsize=5, capthick=1,
#                         label=target_label)
#     # plt.plot(test_rmse[i, :], label=f"Test RMSE {target_label}", linestyle="-", fillstyle='none')
#     plt.legend()
# plt.show()
