from sklearn.preprocessing import MinMaxScaler

from config.param import DATA_DIR, PLOT_DIR
import pandas as pd
from matplotlib import pyplot as plt

from utils.data_loading import csv_to_dataframe

from tensorflow import keras

df = csv_to_dataframe(f"{DATA_DIR}/sessions", "subject_4.csv")

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
    plt.savefig(f'{PLOT_DIR}/parameters_plot.png')
    plt.show()
    plt.close()


show_raw_visualization(df)
