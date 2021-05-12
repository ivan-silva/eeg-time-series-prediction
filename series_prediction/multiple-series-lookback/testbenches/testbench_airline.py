from config.param import DATA_DIR
from generic_multiple_series_lookback import generic_multiple_series_lookback

input_csv_file = f"{DATA_DIR}/airline-passengers.csv"
sel_features = [
    "Passengers",
    "Passengers",
    "Passengers",
    "Passengers",
    "Passengers",
    "Passengers",
]

generic_multiple_series_lookback(
    input_csv_file,
    sel_features,
    csv_sep=",",
    train_split=0.67,
    look_back=15,
    batch_size=1,
    epochs=20,
    verbose=2,
    plot_prefix="airline_recursive_",
    force_retrain=True
)
