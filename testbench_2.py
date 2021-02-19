from generic_multiple_series_lookback import generic_multiple_series_lookback

input_csv_file = "data/sessions_complete/eeg_giorgia.csv"
csv_sep = ","
sel_features = [
    "alpha1",
    "alpha2",
    "beta1",
    "beta2",
    "delta",
    "gamma1",
    "gamma2",
    "theta"
]
train_split = 0.67
look_back = 5
batch_size = 8
epochs = 10
plot_prefix = "eeg_complete_session"
lstm_units = 32

generic_multiple_series_lookback(
    input_csv_file,
    sel_features,
    train_split=train_split,
    look_back=look_back,
    lstm_units=lstm_units,
    batch_size=batch_size,
    epochs=epochs,
    plot_prefix=plot_prefix
)
