from sklearn.preprocessing import MinMaxScaler

from config.param import DATA_DIR, PLOT_DIR
import pandas as pd
from matplotlib import pyplot as plt

from utils.data_loading import csv_to_dataframe
from scipy import signal
import numpy as np


from tensorflow import keras

df = csv_to_dataframe(f"{DATA_DIR}/sessions", "subject_1.csv")

del df['Meditazione']
del df['Attenzione']

# feature_keys = list(df.columns)
titles = feature_keys = ['Alfa1', 'Alfa2', 'Beta1', 'Beta2', 'Delta', 'Gamma1', 'Gamma2', 'Theta']
print(feature_keys)


def smoothing_function(Y):
    return signal.medfilt(Y, kernel_size=3)


# Parameter filtering. We select only interesting parameters. In this case we are removing "Meditazione" and
# "Attenzione" because they are calculated.
print("\nParameter filtering =====================================================================================")
selected_parameters = [0, 1, 2, 3, 4, 5, 6, 7]
print(
    "The selected parameters are:",
    ", ".join([titles[i] for i in selected_parameters]),
)
selected_features = [feature_keys[i] for i in selected_parameters]

# Noise removal
# for j, feature in enumerate(selected_features):
    # df[feature] = smoothing_function(df[feature
# ])

features = df[selected_features]
# Data normalization
print("\nData normalization ======================================================================================")


# Standard score
def standard_score(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


print("Values before normalization", features)
df.plot(x='Session')
plt.tight_layout(pad=3)
plt.savefig(f'{PLOT_DIR}/raw_parameters.png')
plt.show()

colms = df.columns

# features = standard_score(features.values, train_split)
scaler = MinMaxScaler((0, 1))
features = scaler.fit_transform(features)

features = pd.DataFrame(features)
features.columns = selected_features
print("Values after normalization", features)
features.plot()
plt.tight_layout(pad=3)
plt.savefig(f'{PLOT_DIR}/normalized_parameters.png')
plt.show()

# features_sm = pd.DataFrame(smooth_dataset)
# features_sm.columns = selected_features
# features.plot()
# plt.show()
