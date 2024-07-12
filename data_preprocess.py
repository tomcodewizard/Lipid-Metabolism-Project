import numpy as np
import pandas as pd
import os

class DataPreprocess:

    def __init__(self, gene_values_subset: list[str] = None, N : int = 10, data_type : str = 'Concentration', display_progress : bool = False):

        self.gene_values_subset : list[str] = gene_values_subset
        self.N : int = N
        self.data_type : str = data_type
        self.display_progress : bool = display_progress

        possible_data_types = ['Concentration', 'Moment M0', 'Moment M1', 'Moment M2']
        assert self.data_type in possible_data_types, f"data_type must be one of {possible_data_types}"


    def get_pool_dfs(self) -> list[pd.DataFrame]:

        #data_path = os.path.abspath('../../data')
        pool1_path = 'Data/All days pool1.csv'
        pool2_path = 'Data/All days pool2.csv'

        pool1_data = pd.read_csv(pool1_path, low_memory=False)
        pool2_data = pd.read_csv(pool2_path, low_memory=False)

        pool1_df = pd.DataFrame(pool1_data)
        pool2_df = pd.DataFrame(pool2_data)

        pool1_df = pool1_df.fillna('nan')
        pool2_df = pool2_df.fillna('nan')

        pools = [pool1_df, pool2_df]

        return pools
    

    def get_gene_names(self) -> list[str]:

        pools = self.get_pool_dfs()

        pool1_df = pools[0]

        all_gene_values = pool1_df['Gene'].unique()
        all_gene_values = all_gene_values[all_gene_values != 'nan']

        if self.gene_values_subset is None:

            gene_values = all_gene_values
            
        else:
           
            gene_values = self.gene_values_subset

        
        return gene_values


    def average_data(self) -> list[float]:

        volume = 1080 * 1080
        C = 1
        T = 1

        pools = self.get_pool_dfs()
        pool1_df = pools[0]

        gene_values = self.get_gene_names()

        time = [4, 6, 8, 11, 14]

        knockdown_data = []

        for gene_index, g in enumerate(gene_values):

            num_wells = len(pool1_df[pool1_df['Gene'] == g]['Metadata_Well'].unique())
            
            if self.display_progress:

                print(f"{g} : {num_wells} wells")

            gene_data = np.zeros((2, num_wells, 16, 5, 51))

            for pool_index, p in enumerate(pools):

                gene_table = p[p['Gene'] == g]

                wells = gene_table['Metadata_Well'].unique()

                for well_index, w in enumerate(wells):

                    gene_table_well = gene_table[gene_table['Metadata_Well'] == w].iloc[:,:114]

                    for i, t in enumerate(time):

                        df = gene_table_well[gene_table_well['Day'] == t]

                        for f in range(1, 17):

                            field_df = df[df['Metadata_field'] == f]

                            for k in range(1, 52):
                                
                                nk_droplets = field_df[f'Classify_AreaShape_Area_Bin_{k}_NumObjectsPerBin'].sum()

                                if self.data_type == 'Concentration' or self.data_type == 'Moment M0':

                                    gene_data[pool_index, well_index, f - 1, i, k - 1] = nk_droplets / (volume * C)

                                elif self.data_type == 'Moment M1':

                                    gene_data[pool_index, well_index, f - 1, i, k - 1] = k * nk_droplets / (volume * C)

                                elif self.data_type == 'Moment M2':

                                    gene_data[pool_index, well_index, f - 1, i, k - 1] = (k**2) * nk_droplets / (volume * C)

            mean_well_data = np.mean(gene_data, axis=2)
            mean_gene_data = np.mean(mean_well_data, axis=1)
            mean_pool_data = np.mean(mean_gene_data, axis=0)
            mean_overall_data = mean_pool_data[:,:self.N]

            if self.data_type != 'Concentration':

                mean_overall_data = np.sum(mean_overall_data, axis=1)

            knockdown_data.append(mean_overall_data)

        return knockdown_data
    

