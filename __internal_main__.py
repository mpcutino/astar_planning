from ast import literal_eval


from ospa.utils import get_next_state
from ospa.actions import actions
from ospa.flight_state import from_df_format_to_Flight_State, FlightState, fs2dimensional

from heuristics.astar import AStart
from heuristics.utils import distance, plot_from_fs

from data.load_data import landing_data


if __name__ == "__main__":
    astart_is_born = AStart(lambda x: x.cost, distance, actions, get_next_state)

    converters = {'initial state': literal_eval, 'current_state': literal_eval, 'goal_state': literal_eval} 
    landing_df = landing_data(**{'converters': converters})

    sample = landing_df.sample(n=1).iloc[0]
    init_fs = from_df_format_to_Flight_State(sample.initial_state)
    target_fs = from_df_format_to_Flight_State(sample.goal_state)

    b_path = astart_is_born.build_endBased_path(init_fs, target_fs, [], astart_is_born.has_finished)
    plot_from_fs([fs2dimensional(f) for f in b_path], target=fs2dimensional(target_fs))
