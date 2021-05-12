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

filename = "subject_1.csv"
subject_name = filename.replace(".csv", "")
subject_name = subject_name.replace("_", " ")
subject_name = subject_name.capitalize()
df = csv_to_dataframe(f"{DATA_DIR}/sessions", filename)

# feature_keys = list(df.columns)
titles = feature_keys = ['Alfa1', 'Alfa2', 'Beta1', 'Beta2', 'Delta', 'Gamma1', 'Gamma2', 'Theta']
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


def fitting_function(x, a, b, c):
    return a + b * x + c * x * x


def smoothing_function(Y):
    return signal.medfilt(Y, kernel_size=3)


def show_raw_visualization(data):
    time_data = data[index_key]
    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(15, 20), dpi=160, facecolor="w", edgecolor="k"
    )
    fig.suptitle(f"{subject_name}", fontsize=22)
    plt.set_cmap("Paired")
    for i in range(len(feature_keys)):
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

        # Least square fitting
        # Initial guess.
        x0 = np.array([0.0, 0.0, 0.0])
        # sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        (a, b, c), matrix = curve_fit(fitting_function, xdata, ydata, x0)
        yapprox = fitting_function(xdata, a, b, c)

        (a, b, c), matrix = curve_fit(fitting_function, xdata, y_data_s, x0)
        yapprox_s = fitting_function(xdata, a, b, c)

        # Subplot
        ax = axes[i // 2, i % 2]
        ax.set_title(key)

        # Data plot

        ax.plot(xdata, ydata, label="Original", )
        ax.plot(xdata, y_data_s, label="Denoised $k=3$d")
        bbox_style = dict(facecolor='white', alpha=0.5)

        # -------------------------------------------------------
        # Lines
        plot_lines = False

        if plot_lines:
            p = 35
            x_line = [0,78]

            initial_value = ydata[:p].mean()
            final_value = ydata[p:].mean()
            y_line = [initial_value, final_value]

            initial_value_s = y_data_s[:p].mean()
            final_value_s = y_data_s[p:].mean()
            y_line_s = [initial_value_s, final_value_s]

            ax.plot(x_line, y_line, label="Linear $p=35$", )
            ax.plot(x_line, y_line_s, label="Denoised linear ", )

            # Lines - Scatter plot
            x_offset = 2
            y_offset = -2
            x1, y1 = 0, initial_value_s
            x3, y3 = len(yapprox_s) - 1, final_value_s
            ax.scatter(x1, y1, color="red")
            ax.text(x1 + x_offset, y1 + y_offset, f"{float('{:0.2f}'.format(initial_value_s))}", bbox=bbox_style)
            ax.scatter(x3, y3, color="red")
            ax.text(x3 + x_offset, y3 + y_offset, f"{float('{:0.2f}'.format(final_value_s))}", bbox=bbox_style)

        # -------------------------------------------------------
        # Curves
        # Line plot
        plot_curves = True

        if plot_curves:

            # Line plot
            ax.plot(xdata, yapprox, label="Original LSV", )
            ax.plot(xdata, yapprox_s, label="Denoised LSV", )

            # Scatter plot
            initial_value = yapprox_s[0]
            mid_value = yapprox_s[int(len(yapprox_s) // 2)]
            final_value = yapprox_s[len(yapprox_s) - 1]

            x_offset = 2
            y_offset = -2
            x1, y1 = 0, initial_value
            x2, y2 = int(len(yapprox_s) // 2), mid_value
            x3, y3 = len(yapprox_s) - 1, final_value

            ax.scatter(x1, y1, color="red")
            ax.text(x1 + x_offset, y1 + y_offset, f"{float('{:0.2f}'.format(initial_value))}", bbox=bbox_style)
            ax.scatter(x2, y2, color="red")
            ax.text(x2 + x_offset, y2 + y_offset, f"{float('{:0.2f}'.format(mid_value))}", bbox=bbox_style)
            ax.scatter(x3, y3, color="red")
            ax.text(x3 + x_offset, y3 + y_offset, f"{float('{:0.2f}'.format(final_value))}", bbox=bbox_style)



        ax.legend()

    fig.tight_layout(pad=3)
    plt.savefig(f'{plot_dir}{plot_prefix}{subject_name}.png')
    plt.show()
    plt.close()


show_raw_visualization(df)
