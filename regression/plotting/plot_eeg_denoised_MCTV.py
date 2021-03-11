from config.param import DATA_DIR
from regression.plotting.MCTV import parameter as pa

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Dataset configuration
from regression.plotting.MCTV.mctv1d import denoising_1D_MCTV

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
csv_sep = ","
na_values = -1
data_folder = f"{DATA_DIR}/sessions/"
n_features = len(sel_features)
n_files = len(input_csv_files)

# Plotting configuration
plot_prefix = "eeg_regression_"
smoothing_factor = 20

# Denoising
sigma = 100


def smoothing_function(Y):
    n = len(Y)
    lamda = np.sqrt(sigma * n) / 5
    K = 100
    err = 0.001
    alpha = 0.3 / lamda
    para = pa.Parameter(lamda, K, err, alpha)
    # return np.average(params)
    return denoising_1D_MCTV(Y, para)


# All dataset must be the same shape. We use the first dataset shape to initialize data structures and we save it
# to check the subsequent datasets compliancy.
dataframe = pd.read_csv(f"{data_folder}{input_csv_files[0]}", sep=csv_sep, na_values=na_values)
m = dataframe.shape[0]

print(f"Constructing dataset with {n_files} files, {n_features} features, "
      f"smoothing with a factor {smoothing_factor}.")

smooth_dataset = np.zeros(shape=(n_files, n_features, m))

# Smooth dataset generation
for i, input_csv_file in enumerate(input_csv_files):
    # Load each file
    dataframe = pd.read_csv(f"{data_folder}{input_csv_file}", sep=csv_sep, na_values=na_values)

    assert dataframe.shape[0] == m, f"All dataset must be the same length of {m}. The current dataset is " \
                                    f"{dataframe.shape[0]} lines long."

    # For each feature we flatten it in a single value
    for j, feature in enumerate(sel_features):
        col_values = dataframe[feature].values
        col_values = col_values.astype('float32')

        smooth_dataset[i, j, :] = smoothing_function(col_values)
        # smooth_dataset[i, j, :] = col_values

# Plotting
ncols = 2
nrows = int(n_files / ncols)
figsize = (9 * ncols, 6 * nrows)
fig, axes = plt.subplots(
    nrows=nrows, figsize=figsize, ncols=ncols, dpi=160, facecolor="w", edgecolor="k"
)
fig.suptitle(f"Parametri ammorbiditi con fattore {smoothing_factor}")
for i in range(n_files):

    # Plot
    if n_files > 1:
        row = int(i // ncols)
        col = i % ncols
        cur_axes = axes[row, col]
    else:
        cur_axes = axes

    cur_axes.set_title(f"{input_csv_files[i]}")
    cur_axes.set_ylim([0, 200])

    for j in range(n_features):
        cur_axes.plot(smooth_dataset[i, j, :], label=f"{sel_features[j]}", linestyle="-")
    cur_axes.legend()

plt.savefig(f'plots/{plot_prefix}_smoothed_dataset_{smoothing_factor}.png')
plt.show()
plt.close()
