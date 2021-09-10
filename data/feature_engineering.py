from ast import literal_eval

import pandas as pd
from sklearn.model_selection import train_test_split

from data.load_data import load_csv


def transform_df(csv_path):
    df = load_csv(csv_path, **{
        'converters': {"current_state": literal_eval, "target_state": literal_eval},
    })
    df.dropna(inplace=True)
    # assuming u, v, theta, omega, x, z . TODO check!!
    cols = ["uc", "vc", "thetac", "omegac", "xc", "zc"]
    df[cols] = pd.DataFrame(df["current_state"].to_list(), index=df.index)
    cols = ["uf", "vf", "thetaf", "omegaf", "xf", "zf"]
    df[cols] = pd.DataFrame(df["target_state"].to_list(), index=df.index)

    df["u"] = df["uf"] - df["uc"]
    df["v"] = df["vf"] - df["vc"]
    df["theta"] = df["thetaf"] - df["thetac"]
    df["omega"] = df["omegaf"] - df["omegac"]
    df["x"] = df["xf"] - df["xc"]
    df["z"] = df["zf"] - df["zc"]
    # maintain cols
    cols = ["u", "v", "theta", "omega", "x", "z", "action"]
    rm_cols = [c for c in df.columns if c not in cols]
    df.drop(columns=rm_cols, inplace=True)

    categorical_actions = df["action"].astype("category").cat
    df["action_codes"] = categorical_actions.codes

    df.to_csv("filtered.csv", index=False)


def split_train_test(csv_path, **train_split_kwargs):
    df = pd.read_csv(csv_path)
    if 'stratify' in train_split_kwargs:
        train_split_kwargs.pop('stratify')
    df_train, df_test = train_test_split(df, **train_split_kwargs, stratify=df['action_codes'])
    df_train.to_csv("landing_train.csv", index=False)
    df_test.to_csv("landing_test.csv", index=False)


if __name__ == '__main__':
    # transform_df("data.csv")
    split_train_test("filtered.csv", **{'test_size': .3, 'shuffle': True, 'random_state': 40})
