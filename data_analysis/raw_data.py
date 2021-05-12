from sklearn.preprocessing import MinMaxScaler

from config.param import DATA_DIR, PLOT_DIR
import pandas as pd
from matplotlib import pyplot as plt

from utils.data_loading import csv_to_dataframe
from scipy import signal
import numpy as np


from tensorflow import keras

df = csv_to_dataframe(f"{DATA_DIR}/sessions", "subject_2.csv")

del df['Meditazione']
del df['Attenzione']
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

# feature_keys = list(df.columns)
titles = feature_keys = ['Alfa1', 'Alfa2', 'Beta1', 'Beta2', 'Delta', 'Gamma1', 'Gamma2', 'Theta']
print(feature_keys)

time_data = df[index_key]
fig, axes = plt.subplots(
    nrows=4, ncols=2, figsize=(15, 20), dpi=160, facecolor="w", edgecolor="k"
)
for i in range(len(feature_keys)):
    key = feature_keys[i]
    c = colors[i % (len(colors))]
    t_data = df[key]
    t_data.index = time_data
    t_data.head()
    print(t_data)
    ax = t_data.plot(
        ax=axes[i // 2, i % 2],
        # color=c,
        rot=25
    )
    ax.set_title(key, fontsize=20)
    ax.set_ylim([0,240])
fig.tight_layout(pad=3)
plt.savefig(f'{PLOT_DIR}/parameters_plot.png')
plt.show()
plt.close()
