from config.param import DATA_DIR
from generic_multiple_series_lookback import generic_multiple_series_lookback

input_csv_file = f"{DATA_DIR}/household_power_consumption.csv"
csv_sep = ","
sel_features = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
]

generic_multiple_series_lookback(
    input_csv_file,
    sel_features,
    train_split=0.67,
    look_back=1,
    batch_size=1,
    epochs=10,
    plot_prefix="airline_passengers",
    lstm_units=32,
    ncols=2,
    na_values="?"
)
