from ast import literal_eval

import torch
import numpy as np
import pandas as pd

from data.load_data import load_csv

from ospa.flight_state import FlightState

from ml_algorithms.ann.structures.dummy import DirectSNet
from ml_algorithms.ann.training.data_utils import load_train_val_test
from ml_algorithms.ann.validation.utils import get_action_codes, do_one_path


METRICS = ['Cost', 'Time', 'Precision', 'Error', 'V-error', 'P-error']

def get_OSPA_values(ospa_df_path):
    ospa_df = load_csv(ospa_df_path, **{
        'converters': {"current_state": literal_eval, "target_state": literal_eval},
    })
    gb_df = ospa_df.groupby(by='id_trajectory', dropna=True)
    if 'incremental_cost' in ospa_df.columns:
        # because we drop na, the last value in the incremental_cost has the total cost of the path.
        # Then we perform a simple aveage to get the average cost per trajectory.
        aux = gb_df.agg({'incremental_cost': lambda x: x.iloc[-2]})['incremental_cost']
        avg_cost = sum(aux)/len(aux)

        print("Avg. cost: {0}".format(avg_cost))

    if 'target_state' in ospa_df.columns and 'current_state' in ospa_df.columns:
        aux = gb_df.agg({'current_state': lambda x: x.iloc[-1], 'target_state': lambda x: x.iloc[-1]})
        cols = ["uc", "vc", "omegac", "thetac", "xc", "zc"]
        aux[cols] = pd.DataFrame(aux["current_state"].to_list(), index=aux.index)
        cols = ["uf", "vf", "omegaf", "thetaf", "xf", "zf"]
        aux[cols] = pd.DataFrame(aux["target_state"].to_list(), index=aux.index)

        aux["u"] = aux["uf"] - aux["uc"]
        aux["v"] = aux["vf"] - aux["vc"]
        aux["omega"] = aux["omegaf"] - aux["omegac"]
        aux["theta"] = aux["thetaf"] - aux["thetac"]
        aux["x"] = aux["xf"] - aux["xc"]
        aux["z"] = aux["zf"] - aux["zc"]

        avg_precision = sum((aux['x']**2 + aux['z']**2)**.5)/len(aux)
        avg_v_error = sum((aux['u']**2 + aux['v']**2)**.5)/len(aux)
        avg_p_error = sum(aux['theta'])/len(aux)

        print("Avg. precision: {0}, Avg. v-error: {1}, Avg. p-error: {2}".format(
            avg_precision, avg_v_error, avg_p_error))


def get_directsnet_values(model_folder, mlp_train_data_path, mlp_test_data_path, test_data_path_full_format):
    action_col_ = "action_codes"
    continuous_var = ['u', 'v', 'omega', 'theta', 'x', 'z']

    action_codes_dict = get_action_codes(mlp_test_data_path)
    full_format_df = load_csv(test_data_path_full_format,
                              **{'converters': {"current_state": literal_eval, "target_state": literal_eval}})

    gb_df = full_format_df.groupby(by='id_trajectory')
    # get only the first value in the trajectory
    aux_df = gb_df.agg({c: lambda x: x.iloc[0] for c in full_format_df.columns})

    out = load_train_val_test(mlp_train_data_path, mlp_test_data_path, action_col_, continuous_var)
    scaler = out[-1]

    model = DirectSNet.load_model(model_folder)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)

    dict_as_ospa = {
        'current_state': [], 'target_state': [], 'incremental_cost': [], 'id_trajectory': [], 'id_in_seq': []
    }
    for _, r in aux_df.iterrows():
        cs_fs = FlightState.from_jint_data(*r.current_state)
        ts_fs = FlightState.from_jint_data(*r.target_state)

        path = do_one_path(model, cs_fs, ts_fs, action_codes_dict, r.timestep, scaler=scaler, device=device)
        for i, s in enumerate(path):
            dict_as_ospa['current_state'].append([s.u, s.v, s.omega, s.theta, s.x, s.z])
            dict_as_ospa['target_state'].append(r.target_state)
            dict_as_ospa['incremental_cost'].append(s.cost)
            dict_as_ospa['id_trajectory'].append(r.id_trajectory)
            dict_as_ospa['id_in_seq'].append(i)

    pd.DataFrame(dict_as_ospa).to_csv("model_output_data.csv", index=False)


if __name__ == '__main__':
    # get_OSPA_values("../../../data/landing_test.csv")

    get_directsnet_values(
        "../",
        "../../../data/landing_train_mlp_format.csv",
        "../../../data/landing_test_mlp_format.csv",
        "../../../data/landing_test.csv"
    )
