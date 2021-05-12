from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from config.param import DATA_DIR, PLOT_DIR
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from utils.data_loading import csv_to_dataframe

from tensorflow import keras

filenames = [
    "subject_1.csv",
    "subject_2.csv",
    "subject_3.csv",
    "subject_4.csv",
    "subject_5.csv",
    "subject_6.csv"
]

do_denoising = True
do_normalization = False
kernel_size = 3

def denoising_function(Y):
    return signal.medfilt(Y, kernel_size=kernel_size)


fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(15, 20), dpi=160, facecolor="w", edgecolor="k"
)
fig.suptitle(f"Correlazione tra onde EEG, con NR", fontsize=22)

for j, filename in enumerate(filenames):
    subject_name = filename.replace(".csv", "")
    subject_name = subject_name.replace("_", " ")
    subject_name = subject_name.capitalize()
    data = csv_to_dataframe(f"{DATA_DIR}/sessions", filename)

    del data['Meditazione']
    del data['Session']
    del data['Attenzione']

    if do_normalization:
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        columns = data.columns
        data = pd.DataFrame(scaled_data, columns=columns)

    sel_features = data.columns
    if do_denoising:
        for _, feature in enumerate(sel_features):
            col_values = data[feature].values
            col_values = col_values.astype('float32')

            print(f"Column {feature} before denoising:")
            print(col_values)
            col_values = denoising_function(col_values)
            print(f"Column {feature} after denoising:")
            print(col_values)
            data[feature] = col_values


    # Subplot
    ax = axes[j // 2, j % 2]
    im = ax.imshow([[0, 1]])
    ax_divider = make_axes_locatable(ax)
    cax1 = ax_divider.append_axes("right", size="7%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1)

    ax.set_title(subject_name, fontsize=20)
    figax = ax.matshow(data.corr())

    col_names = data.columns
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_yticks(np.arange(len(col_names)))

    ax.set_xticklabels(col_names, fontsize=14, rotation=30)
    ax.set_yticklabels(col_names, fontsize=14)
    # ax.gca().xaxis.tick_bottom()
    # ax.yticks(range(data.shape[1]), data.columns, fontsize=12)

    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
fig.tight_layout(pad=3)
plt.savefig(f'{PLOT_DIR}/feature_correlation_heatmap.png', bbox_inches="tight")
plt.show()
plt.close()
