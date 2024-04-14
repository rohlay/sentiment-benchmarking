"""
Custom CSV Editor:
------------------
This script provides functionality to combine datasets and models from two different folder structures and produces a new folder with the modified CSV files.

Parameters:
-----------
- ID1: The 'original' CSV folder from which the primary data is sourced.
- ID2: The folder that contains the datasets/models we want to copy from.
- ID3: The new folder where the combined CSVs will be stored.
- ADD_MODELS: Dictionary specifying which models from each module should be added.
- ADD_DATASETS: List of datasets to be added.

Instructions:
-------------
1. Set the parameters ID1, ID2, ID3 appropriately.
2. Adjust ADD_MODELS and ADD_DATASETS as per your needs.
3. Run the script.

"""

import os
import pandas as pd
import shutil
from sentiment_analysis_project.scripts.config import RESULTS_DIR

MODULE_KEYS = ['Machine Learning', 'Deep Learning', 'Transformers', 'All']
CATEGORIES = ["classification", "regression"]
STATS = ['mean', 'median', 'max', 'min', 'kfold']

# Parameters
ID1 = "ml_dl_eng_removed_nyt"
ID2 = "ml_nyt"
ID3 = "ml_dl_eng_with_nyt_fixed"
ADD_MODELS = []  # models to be added from id2 to id1
ADD_DATASETS = ["N.Y. Editorial"]  # datasets to be added

def display_changes(affected_paths):
    print("Following paths will be affected:")
    for path in affected_paths:
        print(path)

def merge_data_from_two_folders(id1, id2, id3, add_datasets, add_models):
    # Remove id3 directory if it exists
    if os.path.exists(os.path.join(RESULTS_DIR, id3)):
        shutil.rmtree(os.path.join(RESULTS_DIR, id3))

    # Create a copy of id1 to id3
    shutil.copytree(os.path.join(RESULTS_DIR, id1), os.path.join(RESULTS_DIR, id3))

    for module_key in MODULE_KEYS:
        for category in CATEGORIES:
            for stat in STATS:
                path1 = os.path.join(RESULTS_DIR, id1, module_key, category, stat)
                path2 = os.path.join(RESULTS_DIR, id2, module_key, category, stat)
                path3 = os.path.join(RESULTS_DIR, id3, module_key, category, stat)
                # Check if source folder exists
                if os.path.exists(path1):
                    modify_csv_in_folder(path1, path2, path3, add_datasets, add_models)
                    
def modify_csv_in_folder(path1, path2, path3, add_datasets, add_models):
    for csv_file in os.listdir(path1):
        # index_row: models are on the first row, they're implicitly set as column headers
        # index_col=0: use the first column as the index (row names). The first row automatically becomes the header for the columns
        df1 = pd.read_csv(os.path.join(path1, csv_file), sep=',', index_col=0)
        df2 = pd.read_csv(os.path.join(path2, csv_file), sep=',', index_col=0)

        # Add datasets
        datasets_to_add = df2.loc[add_datasets]
        df1 = pd.concat([df1, datasets_to_add])

        # Add models
        models_to_add = df2[add_models]
        df1 = pd.concat([df1, models_to_add], axis=1)
        
        # Copy to destination
        df1.to_csv(os.path.join(path3, csv_file), sep=',', index=True)


def main():
    merge_data_from_two_folders(ID1, ID2, ID3, ADD_DATASETS, ADD_MODELS)

if __name__ == "__main__":
    main()
