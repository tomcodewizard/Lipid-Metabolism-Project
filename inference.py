from data_preprocess import DataPreprocess
from models import S0_alpha_Model, S0_alpha_theta_Model, S0_alpha_beta_Model
import pandas as pd

class Inference:

    def __init__(self, gene_values_subset : list[str] = None, N : int = 10, data_type : str = 'Concentration', 
                 parameters : list[str] = ['S0', 'alpha', 'theta']):

        self.gene_values_subset = gene_values_subset
        self.N = N
        self.data_type = data_type
        self.parameters = parameters

        data_preprocess = DataPreprocess(self.gene_values_subset, self.N, self.data_type, display_progress=False)
        self.gene_values = data_preprocess.get_gene_names()


    def run_optimisation(self):

        ...


    
    def run_bayesian_inference(self):

        ...