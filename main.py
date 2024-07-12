from data_preprocess import DataPreprocess

concentration_data_preprocess = DataPreprocess(gene_values_subset=None, N=10, data_type='Moment M0', display_progress=True)

concentration_data = concentration_data_preprocess.average_data()

print(concentration_data)