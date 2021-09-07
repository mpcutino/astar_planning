
class AStart:


    def __init__(self, g, h, actions, action_function, obstacles_cmp_func=None):
        self.g = g
        self.h = h
        self.actions = actions
        self.move_func = action_function
        self.cmp_func = obstacles_cmp_func


    def make_one(self, current_pos, target, obstacles=None):
        max_val = self.h(current_pos, target)

        best_action, best_fvalue, best_next = -1, max_val, -1  
        for a in self.actions:
            new_p = self.move_func(current_pos, a, target)
            # if the path has ended, then the output should be computed for the pprevious state
            new_p = current_pos if self.has_finished(new_p) else new_p

            if self.hit_obstacle(new_p, obstacles):
                h_value = max_val
            else: 
                h_value = self.h(new_p, target)
            g_value = self.g(new_p)

            if g_value + h_value < best_fvalue:
                best_fvalue = g_value + h_value
                best_action = a
                best_next = new_p
        return best_action, best_fvalue, best_next

    def build_endBased_path(self, start, end, obstacles, stop_condition):
        b_path = []
        current_state = start
        while not stop_condition(current_state):
            b_path.append(current_state)
            _, _, current_state = self.make_one(current_state, end, obstacles)
        return b_path

    def hit_obstacle(self, state, obstacles):
        if self.cmp_func is None:
            if obstacles is not None and len(obstacles):
                print("Warning!!! There are obstacles but no compare function.")
            return False
        return any(map(lambda x: self.cmp_func(state, x), obstacles))
    

    def has_finished(self, possible_state):
        return type(possible_state) is int
