from math import sqrt, pi, radians


from ospa.constants import Uc, c, k_aero


class FlightState:

    def __init__(self, tail_angle, fa, u, v, theta, omega, x, z):
        self.x = x
        self.z = z
        self.u = u
        self.v = v
        self.theta = theta
        self.omega = omega
        self.tail_angle = tail_angle
        self.fa = fa
        self.cost = 0
    
    def get_info_formatted(self):
        '''
        para obtener la informacion en forma de diccionario
        '''
        # return {
        #     'x': self.x,
        #     'z': self.z,
        #     'u': self.u,
        #     'v': self.v,
        #     'theta': self.theta,
        #     'omega': self.omega,
        #     'tail_angle': self.tail_angle,
        #     'fa': self.fa,
        #     'cost': self.cost,
        # }
        return (self.x, self.z, self.u, self.v, self.theta, self.omega, self.tail_angle, self.fa, self.cost)
    
    @staticmethod
    def order_as_input(input_tuple, tail_angle, fa):
        x, z, u, v, theta, omega = input_tuple
        return FlightState(tail_angle, fa, u, v, theta, omega, x, z)

    def realXvalue(self):
        return self.x * c

    def realZvalue(self):
        return self.z * c

    def Ub_value(self):
        return sqrt(self.v**2 + self.u**2)

    def real_velocity_value(self):
        return self.Ub_value() * Uc

    def increment_cost(self, cost0, P):

        self.cost = cost0 + P
        return self.cost

    def __str__(self):
        return "X: "+str(self.x) + " Z: "+str(self.z) + "\nV: "+str(self.v) + " U: "+str(self.u) + \
               "\nTHETA: " + str(self.theta) + " OMEGA: " + str(self.omega) +"\nTAIL_ANGLE: " + str(self.tail_angle) + \
               " COST: " + str(self.cost)


def from_df_format_to_Flight_State(row_data):
    
    # X axis position (m)
    # Z axis position (m)
    # Velocity in the X axis (m/s)
    # Velocity in the Z axis (m/s)
    # Pitch value (rad)
    # Angular velocity (rad/s)

    # return FlightState.order_as_input(row_data, 0, 0)
    return FlightState(0, 0, row_data[2]/Uc, row_data[3]/Uc, row_data[4], row_data[5], 2*row_data[0]/c, 2*row_data[1]/c)
    # return FlightState(0, 0, row_data[2], row_data[3], row_data[4], row_data[5], row_data[0], row_data[1])
