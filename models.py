import numpy as np
from numba import jit
import pints
from scipy.integrate import odeint

class Models:

    def __init__(self):

        ...

    
class S0_alpha_Model(pints.ForwardModel):

    def __init__(self, initial_conditions, parameters):

        self.initial_conditions = initial_conditions
        self._param_names = ["S0", "alpha"]
        self.parameters = parameters
    

    def get_param_names(self):

        return self._param_names


    def n_parameters(self):

        return 2
    

    def n_outputs(self):

        return len(self.initial_conditions)


    def simulate(self, p, times):

        sol = odeint(self.rhs, self.initial_conditions, times, (p,))
        return sol


    @staticmethod
    @jit
    def rhs(c, t, p):


        def sum1(c, i, alpha, N):

            total = 0

            for k in range(1, N - i + 1):

                total += alpha * c[i - 1] * c[k - 1]

            return total
        
        
        def sum2(c, i, alpha):

            total = 0

            for k in range(1, i):

                total += alpha * c[k - 1] * c[i - k - 1]

            return total
        

        N = len(c)
        S0 = p[0]
        alpha = p[1]

        dc_dt = np.empty(0)

        dc1_dt = S0 - sum1(c, 1, alpha, N)
        dc_dt = np.append(dc_dt, dc1_dt)

        for i in range(2, N + 1):

            dci_dt = -sum1(c, i, alpha, N) + 0.5 * sum2(c, i, alpha)
            dc_dt = np.append(dc_dt, dci_dt)

        return dc_dt
    

class S0_alpha_theta_Model(pints.ForwardModel):

    def __init__(self, initial_conditions, parameters):

        self.initial_conditions = initial_conditions
        self._param_names = ["S0", "alpha", "theta"]
        self.parameters = parameters
    

    def get_param_names(self):

        return self._param_names


    def n_parameters(self):

        return 3
    

    def n_outputs(self):

        return len(self.initial_conditions)


    def simulate(self, p, times):

        sol = odeint(self.rhs, self.initial_conditions, times, (p,))
        return sol


    @staticmethod
    @jit
    def rhs(c, t, p):


        def sum1(c, i, alpha, N):

            total = 0

            for k in range(1, N - i + 1):

                total += alpha * c[i - 1] * c[k - 1]

            return total
        
        
        def sum2(c, i, alpha):

            total = 0

            for k in range(1, i):

                total += alpha * c[k - 1] * c[i - k - 1]

            return total
        

        def S(t, S0):

            return S0

        N = len(c)
        S0 = p[0]
        alpha = p[1]
        theta = p[2]

        dc_dt = np.empty(0)

        dc1_dt = S(t, S0) - sum1(c, 1, alpha, N) - theta * c[0]
        dc_dt = np.append(dc_dt, dc1_dt)

        for i in range(2, N + 1):

            dci_dt = -sum1(c, i, alpha, N) + 0.5 * sum2(c, i, alpha) - theta * c[i - 1] + theta * c[i - 2]
            dc_dt = np.append(dc_dt, dci_dt)

        return dc_dt
    

class S0_alpha_beta_Model(pints.ForwardModel):

    def __init__(self, initial_conditions, parameters):

        self.initial_conditions = initial_conditions
        self._param_names = ["S0", "alpha", "beta"]
        self.parameters = parameters
    

    def get_param_names(self):

        return self._param_names


    def n_parameters(self):

        return 3
    

    def n_outputs(self):

        return len(self.initial_conditions)


    def simulate(self, p, times):

        sol = odeint(self.rhs, self.initial_conditions, times, (p,))
        return sol


    @staticmethod
    @jit
    def rhs(c, t, p):


        def sum1(c, i, alpha, N, beta):

            total = 0

            for k in range(1, N - i + 1):

                total += alpha * c[i - 1] * c[k - 1]
                total -= beta * c[i + k - 1]

            return total
        
        
        def sum2(c, i, alpha, beta):

            total = 0

            for k in range(1, i):

                total += alpha * c[k - 1] * c[i - k - 1]
                total -= beta * c[i - 1]

            return total
        

        def S(t, S0):

            return S0

        N = len(c)
        S0 = p[0]
        alpha = p[1]
        beta = p[2]

        dc_dt = np.empty(0)

        dc1_dt = S(t, S0) - sum1(c, 1, alpha, N, beta)
        dc_dt = np.append(dc_dt, dc1_dt)

        for i in range(2, N + 1):

            dci_dt = -sum1(c, i, alpha, N, beta) + 0.5 * sum2(c, i, alpha, beta)
            dc_dt = np.append(dc_dt, dci_dt)

        return dc_dt