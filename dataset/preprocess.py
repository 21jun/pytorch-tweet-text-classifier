import pandas as pd
from pathlib import Path
from utils.text.clean import normalise_text
from sklearn.model_selection import train_test_split


def preprocess_text(data_path):
    path = Path(data_path)

    train = pd.read_csv(path/"train.csv")
    test = pd.read_csv(path/"test.csv")

    train.drop(columns=['id', 'keyword', 'location'], inplace=True)
    test.drop(columns=['id', 'keyword', 'location'], inplace=True)
    train["text"] = normalise_text(train["text"])
    test["text"] = normalise_text(train["text"])

    train_df, valid_df = train_test_split(train)
    test_df = test

    return {"train": train_df, "valid": valid_df, "test": test_df}
