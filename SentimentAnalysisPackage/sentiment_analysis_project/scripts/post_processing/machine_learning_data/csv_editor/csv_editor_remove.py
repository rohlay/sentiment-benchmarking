import os
import pandas as pd
import shutil
from shutil import copytree, ignore_patterns
from sentiment_analysis_project.scripts.config import ML_METRICS_DIR

MODULE_KEYS = ['Machine Learning', 'Deep Learning', 'Transformers', 'All']
STATS = ['mean', 'median', 'max', 'min']

# Parameters section
IDENTIFIER = "#3"  # for example: "experiment1"
CLASS_OR_REG = "regression"  # either "classification" or "regression"
DEL_DATASETS = []  # replace with the datasets you wish to delete
DEL_MODELS = ["Linear Regression"]  # replace with the models you wish to delete

def modify_df_based_on_datasets_and_models(df, del_datasets, del_models):
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Filter rows based on the first column
    df_copy = df_copy[~df_copy.iloc[:, 0].isin(del_datasets)]
    
    # Drop any models (columns) specified in del_models
    df_copy.drop(columns=del_models, inplace=True, errors='ignore')
    
    return df_copy

def apply_modifications_to_csvs(identifier, class_or_reg, del_datasets, del_models):

    mod_identifier = f"{identifier}_modified"
    source_path = os.path.join(ML_METRICS_DIR, identifier)
    dest_path = os.path.join(ML_METRICS_DIR, mod_identifier)

    # Overwrite destination directory if it exists
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    copytree(source_path, dest_path, ignore=ignore_patterns('*kfold*'))

    for module_key in MODULE_KEYS:
        for stat in STATS:
            path = os.path.join(ML_METRICS_DIR, mod_identifier, module_key, class_or_reg, stat)
            # Check if the directory path exists before processing the CSV files in it.
            if os.path.exists(path):
                for csv_file in os.listdir(path):
                    df = pd.read_csv(os.path.join(path, csv_file))
                    modified_df = modify_df_based_on_datasets_and_models(df, del_datasets, del_models)
                    modified_df.to_csv(os.path.join(path, csv_file), index=False)

def main():
    apply_modifications_to_csvs(IDENTIFIER, CLASS_OR_REG, DEL_DATASETS, DEL_MODELS)

if __name__ == "__main__":
    main()
