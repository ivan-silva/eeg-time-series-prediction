from config.param import DATA_DIR
from generic_multiple_series_lookback import generic_multiple_series_lookback

input_csv_file = f"{DATA_DIR}/sessions/multi_subject_alfa_1.csv"
sel_features = [
    "Subject1, Alfa1",
    "Subject2, Alfa1",
    "Subject3, Alfa1",
    "Subject4, Alfa1",
    "Subject5, Alfa1",
    "Subject6, Alfa1",
]

for i in range(1, 30, 3):
    generic_multiple_series_lookback(
        input_csv_file,
        sel_features,
        csv_sep=",",
        train_split=0.67,
        look_back=i,
        batch_size=6,
        epochs=35,
        verbose=2,
        plot_prefix="eeg_multiple_subjects_",
        force_retrain=True
    )
