from generic_multiple_series_lookback import generic_multiple_series_lookback

input_csv_file = "data/sessions/multi_subject_alfa_1.csv"
csv_sep = ","
sel_features = [
    "Subject1, Alfa1",
    "Subject2, Alfa1",
    "Subject3, Alfa1",
    "Subject4, Alfa1",
    "Subject5, Alfa1",
    "Subject6, Alfa1",
]
train_split = 0.67
look_back = 1
batch_size = 1
epochs = 100
plot_prefix = "eeg_multiple_subjects_"

generic_multiple_series_lookback(
    input_csv_file,
    sel_features,
    csv_sep=",",
    train_split=0.67,
    look_back=1,
    batch_size=1,
    epochs=100,
    verbose=2,
    plot_prefix=plot_prefix
)
