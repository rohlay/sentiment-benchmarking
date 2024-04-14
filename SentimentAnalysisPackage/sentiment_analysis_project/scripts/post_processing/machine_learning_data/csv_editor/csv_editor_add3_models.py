"""
Custom CSV Editor:
------------------
This script provides functionality to combine datasets from different folder structures and produces a new folder with the modified CSV files.

Parameters:
-----------
- ID1: The 'original' CSV folder from which the primary data is sourced.
- ID2: First folder containing the models we want to copy from.
- ID3: Second folder containing the models we want to copy from.
- ID4: The new folder where the combined CSVs will be stored.
- ADD_MODELS_ID2: List of models to be added from ID2.
- ADD_MODELS_ID3: List of models to be added from ID3.

Instructions:
-------------
1. Set the parameters ID1, ID2, ID3, ID4 appropriately.
2. Adjust ADD_MODELS_ID2 and ADD_MODELS_ID3 as per your needs.
3. Run the script.

"""

import os
import pandas as pd
import shutil
from shutil import copytree, ignore_patterns
from sentiment_analysis_project.scripts.config import RESULTS_DIR

MODULE_KEYS = ['Machine Learning', 'Deep Learning', 'Transformers', 'All']
CATEGORIES = ["classification", "regression"]
STATS = ['mean', 'median', 'max', 'min']

ID1 = "metrics_5_1"
ID2 = "metrics_5_2"
#ID3 = "metrics_5_3"
ID4 = "metrics_5"
ADD_MODELS_ID2 = ["CNN", "RNN", "LSTM", "GRU"]
#ADD_MODELS_ID3 = ["Model4"]

def does_folder_exist(id, module_key, category):
    return os.path.exists(os.path.join(RESULTS_DIR, id, module_key, category))

def merge_data_from_folders(source_id, dest_id, add_models):
    for module_key in MODULE_KEYS:
        if os.path.exists(os.path.join(RESULTS_DIR, source_id, module_key)):
            for category in CATEGORIES:
                if does_folder_exist(dest_id, module_key, category) or does_folder_exist(source_id, module_key, category):
                    for stat in STATS:
                        path_source = os.path.join(RESULTS_DIR, source_id, module_key, category, stat)
                        path_dest = os.path.join(RESULTS_DIR, dest_id, module_key, category, stat)
                        merge_csv(path_source, path_dest, add_models)
                        
def merge_csv(path_source, path_dest, add_models):
    if os.path.exists(path_source):
        print(f"Path {path_source} exists.")
        for csv_file in os.listdir(path_source):
            if csv_file.endswith(".csv"):
                source_csv_path = os.path.join(path_source, csv_file)
                dest_csv_path = os.path.join(path_dest, csv_file)
                os.makedirs(os.path.dirname(dest_csv_path), exist_ok=True)

                source_df = pd.read_csv(source_csv_path, index_col=0)
                #print(source_df)
                # Filter only those models that are in the source CSV and were specified to be added
                models_to_add = [model for model in source_df.columns if model in add_models]
                #print(f"\n Adding models...\n{models_to_add}\n")

                if not os.path.exists(dest_csv_path):
                    source_df.to_csv(dest_csv_path)
                else:
                    dest_df = pd.read_csv(dest_csv_path, index_col=0)
                    
                    # Making sure not to add already existing models
                    models_to_add = [model for model in models_to_add if model not in dest_df.columns]
                    merged_df = pd.concat([dest_df, source_df[models_to_add]], axis=1)
                    merged_df.to_csv(dest_csv_path)
                                    
def main():
    if os.path.exists(os.path.join(RESULTS_DIR, ID4)):
        shutil.rmtree(os.path.join(RESULTS_DIR, ID4))
    source_path = os.path.join(RESULTS_DIR, ID1)
    dest_path = os.path.join(RESULTS_DIR, ID4)
    copytree(source_path, dest_path, ignore=ignore_patterns('*kfold*'))
    merge_data_from_folders(ID2, ID4, ADD_MODELS_ID2)
    #merge_data_from_folders(ID3, ID4, ADD_MODELS_ID3)

if __name__ == "__main__":
    main()
