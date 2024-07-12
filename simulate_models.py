import numpy as np
import matplotlib.pyplot as plt
from models import S0_alpha_Model, S0_alpha_theta_Model, S0_alpha_beta_Model

class Simulate_Models:

    def __init__(self, initial_conditions : list[float], N : int = 10, parameters : list[str] = ['S0', 'alpha', 'theta']):

        self.N = N
        self.initial_conditions = initial_conditions
        self.parameters = parameters

        assert len(self.initial_conditions) == self.N, f"Must be (N=){self.N} initial conditions"

        possible_parameters = [['S0', 'alpha'],
                               ['S0', 'alpha', 'theta'],
                               ['S0', 'alpha', 'beta']]
        
        assert self.parameters in possible_parameters, f"Possible parameter sets for models are in {possible_parameters}"

        if self.parameters == possible_parameters[0]:

            self.model = S0_alpha_Model(self.initial_conditions, self.parameters)

        elif self.parameters == possible_parameters[1]:

            self.model = S0_alpha_theta_Model(self.initial_conditions, self.parameters)

        elif self.parameters == possible_parameters[2]:

            self.model = S0_alpha_beta_Model(self.initial_conditions, self.parameters)


    def concentration_time_graph(self):

        ...