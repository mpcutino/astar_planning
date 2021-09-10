import numpy as np
import pandas as pd


def get_desired_vars(df: pd.DataFrame, state_column, action_column):
    states = np.array([np.array(r) for r in df[state_column].values])
    categorical_actions = df[action_column].astype("category").cat
    actions = categorical_actions.codes.values
    action_categories = categorical_actions.categories.values

    return {'states': states, 'actions': actions, 'categories': action_categories}
