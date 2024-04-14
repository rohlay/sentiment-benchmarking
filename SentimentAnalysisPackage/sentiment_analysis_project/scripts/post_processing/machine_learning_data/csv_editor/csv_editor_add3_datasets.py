"""
Custom CSV Editor:
------------------
This script provides functionality to combine datasets from different folder structures and produces a new folder with the modified CSV files.

Parameters:
-----------
- ID1: The 'original' CSV folder from which the primary data is sourced.
- ID2: First folder containing the datasets we want to copy from.
- ID3: Second folder containing the datasets we want to copy from.
- ID4: The new folder where the combined CSVs will be stored.
- ADD_DATASETS_ID2: List of datasets to be added from ID2.
- ADD_DATASETS_ID3: List of datasets to be added from ID3.
- Assumtion: All models (columns) are the same in csv files.

Instructions:
-------------
1. Set the parameters ID1, ID2, ID3, ID4 appropriately.
2. Adjust ADD_DATASETS_ID2 and ADD_DATASETS_ID3 as per your needs.
3. Run the script.

"""
import os
import pandas as pd
import shutil
from shutil import copytree, ignore_patterns
from sentiment_analysis_project.scripts.config import ML_METRICS_DIR

MODULE_KEYS = ['Machine Learning', 'Deep Learning', 'Transformers', 'All']
CATEGORIES = ["classification", "regression"]
STATS = ['mean', 'median', 'max', 'min']

ID1 = "#6"
ID2 = "#7"
#ID3 = "metrics_tf_esp_6"
ID4 = "#1&2"
ADD_DATASETS_ID2 = ["SST2_2" ,"SST5_2"]
#ADD_DATASETS_ID3 = ["Tweets"]

def does_folder_exist(id, module_key, category):
    return os.path.exists(os.path.join(ML_METRICS_DIR, id, module_key, category))

def merge_data_from_folders(source_id, dest_id, add_datasets):
    for module_key in MODULE_KEYS:
        if os.path.exists(os.path.join(ML_METRICS_DIR, source_id, module_key)):
            for category in CATEGORIES:
                # Check if the dest folder exists for the category or if the source folder exists for the category
                if does_folder_exist(dest_id, module_key, category) or does_folder_exist(source_id, module_key, category):
                    for stat in STATS:
                        path_source = os.path.join(ML_METRICS_DIR, source_id, module_key, category, stat)
                        path_dest = os.path.join(ML_METRICS_DIR, dest_id, module_key, category, stat)
                        if os.path.exists(path_source):
                            for csv_file in os.listdir(path_source):
                                if csv_file.endswith(".csv"):
                                    source_csv_path = os.path.join(path_source, csv_file)
                                    dest_csv_path = os.path.join(path_dest, csv_file)
                                    
                                    # Ensure destination path is created if it doesn't exist
                                    os.makedirs(os.path.dirname(dest_csv_path), exist_ok=True)

                                    source_df = pd.read_csv(source_csv_path, index_col=0)
                                    source_df = source_df.loc[source_df.index.intersection(add_datasets)]
                                    if not os.path.exists(dest_csv_path):
                                        source_df.to_csv(dest_csv_path)
                                    else:
                                        dest_df = pd.read_csv(dest_csv_path, index_col=0)
                                        merged_df = pd.concat([dest_df, source_df])
                                        merged_df.to_csv(dest_csv_path)
            
def main():
    if os.path.exists(os.path.join(ML_METRICS_DIR, ID4)):
        shutil.rmtree(os.path.join(ML_METRICS_DIR, ID4))
    source_path = os.path.join(ML_METRICS_DIR, ID1)
    dest_path = os.path.join(ML_METRICS_DIR, ID4)
    copytree(source_path, dest_path, ignore=ignore_patterns('*kfold*')) # start with ID4 equal to ID1
    merge_data_from_folders(ID2, ID4, ADD_DATASETS_ID2)
    #merge_data_from_folders(ID3, ID4, ADD_DATASETS_ID3)

if __name__ == "__main__":
    main()
