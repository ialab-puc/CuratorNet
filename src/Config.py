import os

dir_path = os.path.dirname(os.path.abspath(__file__))

Config = {
    "SALES_FILE": os.path.join(dir_path, '..', 'data', 'purchases.csv'),
    "ARTWORKS_FILE": os.path.join(dir_path, '..','data', 'inventory.csv'),
    "EXPERIMENT_RESULTS_DIRPATH": os.path.join(dir_path, '..', 'results'),
}