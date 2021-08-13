import pandas as pd


def landing_data(**kwargs):
    return load_csv("data/OSPA_landing_training-test_data.csv", **kwargs)


def load_csv(p, **kwargs):
    df = pd.read_csv(p, **kwargs)
    c = [substitute_space(i, '_') for i in df.columns]
    df.columns = c
    return df


def substitute_space(name_, char_):
    return char_.join(name_.split())
