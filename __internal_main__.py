from ast import literal_eval


from ospa.utils import get_next_state
from ospa.actions import actions
from ospa.flight_state import from_df_format_to_Flight_State, FlightState

from heuristics.astar import AStart
from heuristics.utils import distance

from data.load_data import landing_data


if __name__ == "__main__":
    astart_is_born = AStart(lambda x: x.cost, distance, actions, get_next_state)

    converters = {'initial state': literal_eval, 'current_state': literal_eval, 'goal_state': literal_eval} 
    landing_df = landing_data(**{'converters': converters})

    sample = landing_df.iloc[0]
    init_fs = from_df_format_to_Flight_State(sample.initial_state)
    target_fs = from_df_format_to_Flight_State(sample.goal_state)

    from ospa.constants import Uc, c
    s_0 = FlightState(0, 0, 2 / Uc, 1 / Uc, 0, 0, 0, 10 * 2 / c)
    s_f = FlightState(0, 0, 0, 0, 0, 0, 20 * 2 / c, 0)
    init_fs, target_fs = s_0, s_f

    print(init_fs)
    print(target_fs)

    # initi_fs = FlightState(0, 0, 1, 0, 0, 0, 0, 0)

    best_action, best_fvalue, best_next = astart_is_born.make_one(init_fs, target_fs)

    print(best_action)
    print(best_fvalue)
    print(best_next)

    # b_path = astart_is_born.build_endBased_path(init_fs, target_fs, [])
    # print(b_path)
