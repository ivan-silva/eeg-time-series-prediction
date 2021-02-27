from generic_multiple_series_lookback import generic_multiple_series_lookback

input_csv_file = "data/airline-passengers.csv"
sel_features = [
    "Passengers"
]

generic_multiple_series_lookback(
    input_csv_file,
    sel_features,
    train_split=0.67,
    look_back=15,
    batch_size=1,
    epochs=40,
    plot_prefix="airline_passengers",
    lstm_units=32,
    ncols=1,
    force_retrain=True
)
