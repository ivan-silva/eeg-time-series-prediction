import statistics

from keras import regularizers
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import math
from sklearn.metrics import mean_squared_error
from scipy import signal
from keras.layers import Conv1D, Flatten, Conv2D, Dropout

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
from config.param import DATA_DIR, PLOT_DIR

plot_prefix = "lsv_denoised_"
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
    "Hamilton",
    "STAI-1",
    "STAI-2",
    "SPM-Age",
    "SPM-Scolar",
    # "Gender"
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
epochs = 50
verbose = 0

smoothing_factor = 35

# Denoising
do_denoising = True
kernel_size = 3

# LSV
x0 = np.array([0.0, 0.0, 0.0])
# sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Gaussian data augmentation
do_data_augmentation = False
# sigma = [0, 1, 2, 4, 8, 50, 500]
sigma = 1
n = 10


def fitting_function(x, a, b, c):
    return a + b * x + c * x * x


def denoising_function(Y):
    return signal.medfilt(Y, kernel_size=kernel_size)
    # return Y


def noise(X, y, n, sigma):
    _X = X.copy()
    _y = y.copy()
    for _ in range(n):
        X = np.r_[X, _X + np.random.randn(*_X.shape) * sigma]
        y = np.r_[y, _y]
    return X, y


# Set targets and initial targets
initial_targets = pd.read_csv(f'{data_dir}initial_targets.csv')
initial_targets.head()
targets = pd.read_csv(f'{data_dir}targets.csv')
targets.head()

# All files must be the same shape. We use the first dataset shape to initialize data structures and we save it
# to check the subsequent datasets compliancy.
dataframe = pd.read_csv(f"{data_dir}{input_csv_files[0]}", sep=csv_sep, na_values=na_values)
m = dataframe.shape[0]
initial_values = np.zeros(shape=(n_subjects, n_features))
mid_values = np.zeros(shape=(n_subjects, n_features))
final_values = np.zeros(shape=(n_subjects, n_features))

# Dataset generation (without targets)
print(f"Constructing dataset with {n_subjects} files, {n_features} features")

for i, input_csv_file in enumerate(input_csv_files):
    # Load each file
    dataframe = pd.read_csv(f"{data_dir}{input_csv_file}", sep=csv_sep, na_values=na_values)

    assert dataframe.shape[0] == m, f"All dataset must be the same length of {m}. The current dataset is " \
                                    f"{dataframe.shape[0]} lines long."

    for j, feature in enumerate(sel_features):
        col_values = dataframe[feature].values
        col_index = dataframe[feature].index
        col_values = col_values.astype('float32')

        if do_denoising:
            print(f"Column {feature} before denoising:")
            print(col_values)
            col_values = denoising_function(col_values)
            print(f"Column {feature} after denoising:")
            print(col_values)

        xdata = col_index
        ydata = col_values

        (a, b, c), matrix = curve_fit(fitting_function, xdata, dataframe[feature].values, x0)
        yapprox = fitting_function(xdata, a, b, c)

        (a, b, c), matrix = curve_fit(fitting_function, xdata, ydata, x0)
        yapprox_s = fitting_function(xdata, a, b, c)

        initial_values[i, j] = yapprox_s[0]
        mid_values[i, j] = yapprox_s[int(len(yapprox_s) // 2)]
        final_values[i, j] = yapprox_s[len(yapprox_s) - 1]

        # Debug plot for correctness check
        # subject_name = input_csv_file.replace(".csv", "")
        # subject_name = subject_name.replace("_", " ")
        # subject_name = subject_name.capitalize()
        # plt.title(f"{subject_name}, {feature}")
        # plt.plot(xdata, dataframe[feature].values, label="Original")
        # plt.plot(xdata, col_values, label="Denoised")
        # plt.plot(xdata, yapprox, label="Original LSV")
        # plt.plot(xdata, yapprox_s, label="Denoised LSV")
        # x_offset = 2
        # y_offset = -2
        # plt.scatter(0, initial_values[i, j])
        # plt.text(0+x_offset, initial_values[i, j]+y_offset, f"{float('{:0.2f}'.format(initial_values[i, j]))}",
        #          bbox=dict(facecolor='white', alpha=0.5))
        # plt.scatter(int(len(yapprox_s) // 2), mid_values[i, j])
        # plt.text(int(len(yapprox_s) // 2)+x_offset, mid_values[i, j]+y_offset, f"{float('{:0.2f}'.format(mid_values[i, j]))}",
        #          bbox=dict(facecolor='white', alpha=0.5))
        # plt.scatter(len(yapprox_s) - 1, final_values[i, j])
        # plt.text(len(yapprox_s) - 1+x_offset, final_values[i, j]+y_offset, f"{float('{:0.2f}'.format(final_values[i, j]))}",
        #          bbox=dict(facecolor='white', alpha=0.5))
        # plt.legend()
        # plt.savefig(f'{plot_dir}{plot_prefix}{subject_name}_{feature}.png')
        # plt.show()

first_features_names = list(map(lambda feature_name: f"{feature_name}_start", sel_features))
mid_features_names = list(map(lambda feature_name: f"{feature_name}_mid", sel_features))
last_feature_names = list(map(lambda feature_name: f"{feature_name}_end", sel_features))
s_e_dataframe = pd.DataFrame(
    data=np.hstack((initial_values, mid_values, final_values)),
    columns=np.hstack((first_features_names, mid_features_names, last_feature_names))
)
print(s_e_dataframe.head())

input_shape = n_features * 3 + 1

# Output dataset
predictions = np.zeros((n_subjects, n_targets))
train_rmse = np.zeros((n_subjects, n_targets))
train_mae = np.zeros((n_subjects, n_targets))

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
    train_dataframe = pd.DataFrame(
        data=np.hstack((s_e_dataframe.values, np.expand_dims(initial_targets[target_label], axis=1))),
        columns=np.hstack((s_e_dataframe.columns, f"{target_label}_start"))
    )
    print("Complete dataset for feature")
    print(train_dataframe)

    # Specific train set for n-1 subjects with 1 subject as validation
    for j in range(n_subjects):
        print(f"-------------------------------------------------------------------------------")
        print(f"{target_label} dataset for subject {j + 1}")

        # Train set generation with index != j
        X_train = train_dataframe.loc[train_dataframe.index != j].values
        y_train = targets[target_label].loc[targets.index != j].values
        X_train, y_train = noise(X_train, y_train, n, sigma)
        y_train = np.expand_dims(y_train, axis=1)

        # Validation set generation with index == j
        X_val = train_dataframe.loc[train_dataframe.index == j].values
        y_val = targets[target_label].loc[targets.index == j].values
        # X_val, y_val = noise(X_val, y_val, n, sigma)
        y_val = np.expand_dims(y_val, axis=1)

        print(f"Train X: {X_train.shape}")
        print(pd.DataFrame(X_train, columns=train_dataframe.columns))
        print(f"Train y:")
        print(y_train)

        # Train
        # print(f"Replay = {r}")
        # Model definition
        model = Sequential()
        model.add(Dense(128,
                        activation="relu",
                        input_shape=input_shape,
                        activity_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.50))
        model.add(Dense(128,
                        activation="relu",
                        activity_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.50))
        model.add(Dense(1, activation="relu"))
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=verbose)

        colors = ["SteelBlue", "SkyBlue", "Brown", "IndianRed"]
        def visualize_loss(loss_history):
            loss = loss_history.history["loss"]
            val_loss = loss_history.history["val_loss"]
            epochs = range(len(loss))
            plt.figure()
            plt.ylim([0, 50])
            plt.title(f"Training loss and validation loss")
            plt.plot(epochs, loss, colors[0], label="Training loss")
            plt.plot(epochs, val_loss, colors[2], label="Validation loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{PLOT_DIR}/{plot_prefix}_loss_val_loss.png')
            plt.show()
            plt.close()


        # visualize_loss(history)

        # Predict
        prediction = model.predict(X_val)
        print(f"Predicted {target_label}={prediction}. Real value={y_val}. Error={y_val - prediction}")
        predictions[j, i] = prediction
        train_predictions = model.predict(X_train)
        train_rmse[j, i] = math.sqrt(mean_squared_error(y_train, train_predictions))
        train_mae[j, i] = mean_absolute_error(y_train, train_predictions)

        # # Validation errors
        # for t, target in enumerate(target_labels):
        #     test_rmse_i[r, t] = math.sqrt(mean_squared_error(targets[target], prediction))
        #     test_mae_i[r, t] = mean_absolute_error(targets[target], predictions[target])

    output_dataframe: pd.DataFrame = train_dataframe.copy()
    output_dataframe[f"{target_label}_END"] = targets[target_label].values
    output_dataframe[f"{target_label}_PREDICTION"] = predictions[:, i]
    output_dataframe[f"{target_label}_ERR"] = np.diff((targets[target_label].values, predictions[:, i]), axis=0).T
    output_dataframe.to_csv(f"{DATA_DIR}/output/lsv_complete_train_dataset_{target_label}.csv")

predictions = pd.DataFrame(predictions, columns=target_labels)
print("Initial values")
print(initial_targets)
print("Predictions")
print(predictions)
print("Real final values")
print(targets)

# Trend prediction
# trend_real = np.array(initial_targets) < np.array(targets.values)
# trend_predicted = np.array(initial_targets) < np.array(predictions.values)
# trend_correct = trend_real == trend_predicted
# trend_percentage = 100/(trend_real.shape[0] * trend_real.shape[1]) * np.sum(trend_correct)
# print(f"Trend prediction {trend_percentage}% correct")
# print(trend_correct)

# Target errors
rmse = np.zeros(n_targets)
mae = np.zeros(n_targets)
for i, target in enumerate(target_labels):
    rmse[i] = math.sqrt(mean_squared_error(targets[target], predictions[target]))
    mae[i] = mean_absolute_error(targets[target], predictions[target])
print("Target RMSE")
print(rmse)
print("Target MAE")
print(mae)

df = pd.DataFrame({
    "Train RMSE": train_rmse.mean(axis=0),
    "Train MAE": train_mae.mean(axis=0),
    "Validation RMSE": rmse,
    "Validation MAE": mae,
}, index=target_labels)

colors = ["SteelBlue", "SkyBlue", "Brown", "IndianRed"]
ax = df.plot.bar(color=colors, rot=0, title=f"CNN LSV, {epochs} epochs")
ax.set_xlabel("Feature")
ax.set_xticklabels(target_labels, rotation=45)
plt.ylim([0, 50])
plt.tight_layout(pad=3)
plt.savefig(f'{plot_dir}{plot_prefix}_errors_{epochs}.png', bbox_inches="tight")
plt.show()

# # Subject errors
# train_rmse_s = np.zeros(n_subjects)
# train_mae_s = np.zeros(n_subjects)
# test_rmse_s = np.zeros(n_subjects)
# test_mae_s = np.zeros(n_subjects)
# for i in range(n_subjects):
#     train_rmse_s[i] = math.sqrt(mean_squared_error(targets.iloc[i].values, predictions.iloc[i].values))
#     train_mae_s[i] = mean_absolute_error(targets.iloc[i].values, predictions.iloc[i].values)
#     test_rmse_s[i] = math.sqrt(mean_squared_error(targets.iloc[i].values, predictions.iloc[i].values))
#     test_mae_s[i] = mean_absolute_error(targets.iloc[i].values, predictions.iloc[i].values)
# print("Subject RMSE")
# print(test_rmse_s)
# print("Subject MAE")
# print(test_mae_s)
#
# df = pd.DataFrame({
#     "Train RMSE": train_rmse_s,
#     "Train MAE": train_rmse_s,
#     "Validation RMSE": test_rmse_s,
#     "Validation MAE": test_mae_s,
# })
# ax = df.plot.bar(color=["SkyBlue", "IndianRed", "Brown"], rot=0, title=f"Subject errors, {epochs} epochs")
# ax.set_xlabel("Feature")
# plt.ylim([0, 25])
# plt.tight_layout()
# plt.savefig(f'{plot_dir}{plot_prefix}_subject_errors_{epochs}.png', bbox_inches="tight")
# plt.show()

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
