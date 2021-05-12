from sklearn.preprocessing import MinMaxScaler
import numpy as np
from config.param import DATA_DIR, PLOT_DIR
import pandas as pd
from matplotlib import pyplot as plt

from utils.data_loading import csv_to_dataframe
from scipy import signal
from scipy.optimize import curve_fit

from tensorflow import keras

plot_prefix = "lsv_denoised_"
data_dir = f'{DATA_DIR}/sessions/'
plot_dir = f'{PLOT_DIR}/'

filenames = [
    "subject_1.csv",
    "subject_2.csv",
    "subject_3.csv",
    "subject_4.csv",
    "subject_5.csv",
    "subject_6.csv"
]

# feature_keys = list(df.columns)
titles = feature_keys = ['Delta', 'Theta', 'Alfa1', 'Alfa2', 'Beta1', 'Beta2', 'Gamma2', 'Gamma1']
print(feature_keys)

colors = [
    "navy",
    "crimson",
    "orange",
    "black",
    "darkgreen",
    "purple",
    "darkgrey",
    "orangered",
    "cyan",
]

linestyles = [
    "solid",
    "dashed",
    "dashdot",
    "solid",
    "dashed",
    "dashdot",
    "solid",
    "dashed",
    "dashdot",
]
index_key = "Session"
n_features = len(feature_keys)
n_subjects = len(filenames)


def fitting_function(x, a, b, c):
    return a + b * x + c * x * x


def smoothing_function(Y):
    return signal.medfilt(Y, kernel_size=3)


yapprox_s_all = np.zeros((n_features, n_subjects, 78))
for i, feature_key in enumerate(feature_keys):
    # fig, axes = plt.subplots(
    #     nrows=3, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    # )
    # fig.suptitle(f"{feature_key}", fontsize=22)
    # plt.set_cmap("Paired")

    yapprox_avg_i = np.array((n_subjects, 78))
    for j, filename in enumerate(filenames):
        subject_name = filename.replace(".csv", "")
        subject_name = subject_name.replace("_", " ")
        subject_name = subject_name.capitalize()
        data = csv_to_dataframe(f"{DATA_DIR}/sessions", filename)
        time_data = data[index_key]

        key = feature_keys[i]
        color = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        print(t_data)

        # Denoising
        xdata = t_data.index
        ydata = t_data.values
        y_data_s = smoothing_function(ydata)
        # y_data_s = ydata

        # Least square fitting
        # Initial guess.
        x0 = np.array([0.0, 0.0, 0.0])
        # sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        (a, b, c), matrix = curve_fit(fitting_function, xdata, ydata, x0)
        yapprox = fitting_function(xdata, a, b, c)

        (a, b, c), matrix = curve_fit(fitting_function, xdata, y_data_s, x0)
        yapprox_s = fitting_function(xdata, a, b, c)

        yapprox_s_all[i, j, :] = yapprox_s

        # # Subplot
        # ax = axes[j // 2, j % 2]
        # ax.set_title(subject_name)
        #
        # # Line plot
        # ax.plot(xdata, ydata, label="Original", )
        # ax.plot(xdata, y_data_s, label="Denoised", )
        # ax.plot(xdata, yapprox, label="Original LSV", )
        # ax.plot(xdata, yapprox_s, label="Denoised LSV", )
        #
        # # Scatter plot
        # initial_value = yapprox_s[0]
        # mid_value = yapprox_s[int(len(yapprox_s) // 2)]
        # final_value = yapprox_s[len(yapprox_s) - 1]
        #
        # x_offset = 2
        # y_offset = -2
        # x1, y1 = 0, initial_value
        # x2, y2 = int(len(yapprox_s) // 2), mid_value
        # x3, y3 = len(yapprox_s) - 1, final_value
        # bbox_style = dict(facecolor='white', alpha=0.5)
        #
        # ax.scatter(x1, y1, color="red")
        # ax.text(x1 + x_offset, y1 + y_offset, f"{float('{:0.2f}'.format(initial_value))}", bbox=bbox_style)
        # ax.scatter(x2, y2, color="red")
        # ax.text(x2 + x_offset, y2 + y_offset, f"{float('{:0.2f}'.format(mid_value))}", bbox=bbox_style)
        # ax.scatter(x3, y3, color="red")
        # ax.text(x3 + x_offset, y3 + y_offset, f"{float('{:0.2f}'.format(final_value))}", bbox=bbox_style)
        #
        # ax.legend()

    # plt.tight_layout()
    # plt.savefig(f'{plot_dir}{plot_prefix}{feature_key}.png')
    # plt.show()
    # plt.close()
yapprox_s_avg = yapprox_s_all.mean(axis=1)

plt.figure(figsize=(10, 10))
plt.grid(True, 'major', color='grey', linestyle=':')
plt.title("Onde medie, minimi quadrati, con NR")
plt.yticks(np.arange(0, 140, 5))
plt.xticks((1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 78))
# plt.minorticks_on()
# plt.grid(True, 'minor', 'y', color="lavender", linestyle=':')
# plt.margins(x=.5, y=.2)
# plt.margins(x=.08)

values = np.zeros((8, 3))
for i, feature_key in enumerate(feature_keys):
    yapprox_s_avg = yapprox_s_all.mean(axis=1)
    yapprox_s = yapprox_s_avg[i, :]
    xdata = np.arange(1, 79, 1)
    # Line plot
    # ax.plot(xdata, yapprox, label="Original LSV", )

    # Scatter plot
    initial_value = yapprox_s[0]
    mid_value = yapprox_s[int(len(yapprox_s) // 2)]
    final_value = yapprox_s[len(yapprox_s) - 1]

    values[i,:] = (initial_value, mid_value, final_value)

    x_offset = 2
    y_offset = -2
    x1, y1 = 1, initial_value
    x2, y2 = int(len(yapprox_s) // 2), mid_value
    x3, y3 = len(yapprox_s), final_value
    bbox_style = dict(facecolor='white', alpha=0.5)

    plt.scatter(x1, y1, color=colors[i])
    plt.scatter(x2, y2, color=colors[i])
    plt.scatter(x3, y3, color=colors[i])
    plt.text(x1 + x_offset, y1 + y_offset, f"{float('{:0.2f}'.format(initial_value))}", bbox=bbox_style)
    plt.text(x2 + x_offset, y2 + y_offset, f"{float('{:0.2f}'.format(mid_value))}", bbox=bbox_style)
    plt.text(x3 + -6, y3 + y_offset, f"{float('{:0.2f}'.format(final_value))}", bbox=bbox_style)

    plt.plot(xdata, yapprox_s, label=feature_key, color=colors[i], linestyle=linestyles[i])
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

plt.tight_layout()
plt.set_cmap("Paired")
plt.savefig(f'{plot_dir}{plot_prefix}{feature_key}_mean.png')
plt.show()
plt.close()
print(values)
