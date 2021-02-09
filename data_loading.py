import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def csv_to_dataframe(path: str, filename: str):

    # TODO: extract configuration variables
    separator = ","

    abs_path = os.path.join(path, filename)
    print(f"Loading dataset from {abs_path}")
    df = pd.read_csv(abs_path, sep=separator, na_values=-1)
    print(f"Dataset loaded and containing {len(df)} rows")

    return df