from config.param import DATA_DIR
from generic_multiple_series_lookback import generic_multiple_series_lookback


input_csv_file = f"{DATA_DIR}/sessions_complete/eeg_subject_4.csv"
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

for i in range(1, 110, 10):
    train_split = 0.67
    look_back = i
    batch_size = 64
    epochs = 15
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
