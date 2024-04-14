"""
TODO List:
    * FIX: If DL and Tf have different epochs. Error saving arrays of save length to csv
      Currently, only works if epochs is same
    * Save model file. Save best model.
    * Hyperparameters
    * Move merge_results to postprocessing
    * Create spearate module for Times
    * Refactor modules: all_models, machine_learning, deep_learniing, transformer_pretrained
    For example,
    - Avoid loading GloVe file multiple times (once is enough)
    - Fix the transformer models path (overly complicated). It loads every iter.
    - Add model definitions to the module (remove all_models.py)
    - etc.

"""


"""
Author: Rohan Laycock

"""

import shutil
import json
import os
import sys
import time
import glob
import math
from collections import defaultdict
from contextlib import contextmanager
import csv
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

# Importing modules from models directory
from sentiment_analysis_project.scripts.main_analysis.models.machine_learning import MachineLearningModule
from sentiment_analysis_project.scripts.main_analysis.models.deep_learning import DeepLearningModule
from sentiment_analysis_project.scripts.main_analysis.models.transformer_pretrained import TransformersModule
from sentiment_analysis_project.scripts.main_analysis.models.all_models import AllModels

# Importing modules from evaluation directory
from sentiment_analysis_project.scripts.main_analysis.metrics.metrics_calculator import MetricsCalculator

# Importing config module
from sentiment_analysis_project.scripts.config_dir import CONFIG_ML
from sentiment_analysis_project.scripts.config_dir import DATA_PROCESSED
from sentiment_analysis_project.scripts.config_dir import OUTPUTS_DIR

class DualOutputStream:
    def __init__(self, file_path, stdout):
        self.file = open(file_path, 'w')
        self.stdout = stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        # This flush method is needed for compatibility with certain features like tqdm
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

class ModelsEvaluation:
    def __init__(self):
        #---------------------------
        # Parameters defined by user
        #---------------------------
        with open(CONFIG_ML, 'r') as f:
            config = json.load(f)
        evaluation_config = config['models_evaluation']
        self.k = evaluation_config['k']
        self.tune = evaluation_config['tune']
        self.save_models = evaluation_config['save_models']
        self.save_log = evaluation_config["save_log"]
        self.regression_threshold = evaluation_config["regression_threshold"]
        self.save_history_performance = evaluation_config["save_history_performance"]
        #-------------------------------------------------------------
        # Important Parameter Configuration Depending On Problem Type
        # These parameters are automatically detected
        #--------------------------------------------------------------
        self.is_continuous = None
        self.is_multiclass = None
        self.num_classes = None
        self.loss = None

        #---------------------------------------------------------------
        # For Hyperparameter Tuning and Saving the Best Model File.
        #---------------------------------------------------------------
        self.tune_method = 'random_search'
        self.eval_metric = None
        self.current_best_score = None
        self.best_model = None
        self.metric_criterion = {
            'binary': 'accuracy_score', 
            'multiclass': 'accuracy_score', 
            'continuous': 'root_mean_squared_error'
        }
        
        # Initialize the history dicts attribute as a nested defaultdict
        # if a key does not exist at any level, a new empty dict is created
        self.history_folds = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.history_mean = defaultdict(lambda: defaultdict(dict))

        #---------------------------------------------
        # Get the experiment ID and create directories 
        #---------------------------------------------
        self.expID = self.get_new_id()
        self.EXP_DIR = None
        self.create_output_dirs()

        #----------------------------------------------------------------
        # Logging
        # - A copy of the json file used when running the experiment.
        # - A log text file where the terminal outputed is dumped
        #----------------------------------------------------------------
        if self.save_log:
            log_file_path = os.path.join(self.EXP_DIR, f"{self.expID}.txt")
            self.log_file = self.setup_logging(log_file_path) # Setup the log file for terminal outputs
            # Saving a copy of the config_ml.json inside the ID directory
            with open(CONFIG_ML, 'r') as source_file:
                data = source_file.read()
                with open(os.path.join(self.EXP_DIR, 'config_ml.json'), 'w') as target_file:
                    target_file.write(data)

    ########################################################################################################
    ########################################################################################################
    
    ### -------------------------------------------
    ### Experiment Identifier and Directories Setup
    ### -------------------------------------------
    """
    A new identifier is created for every experiment.
    """
    def get_new_id(self):
        # Create /output_results, if it doesn't exist.
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        # Get today's date in the format YYYYMMDD
        today = time.strftime("%Y_%m_%d")
        # Get all directories that start with today's date
        dirs_today = sorted([os.path.basename(path) for path in glob.glob(f"{OUTPUTS_DIR}/{today}_*")])
        # Determine new experiment number based on existing directories
        if dirs_today:
            last_num_today = int(dirs_today[-1].split('_')[-1])
            new_num_today = last_num_today + 1
        else:
            new_num_today = 1

        expID = f"{today}_{new_num_today}"
        return expID
    """
    In each experiment, a folder is created with the experiment identifier.
    All the outputs are stored in this folder.
    """
    def create_output_dirs(self):

        # Experiment ID
        self.EXP_DIR = os.path.join(OUTPUTS_DIR, f'{self.expID}')

        # Outputs/results per experiment
        self.SAVED_MODELS_DIR = os.path.join(self.EXP_DIR, 'saved_models')
        self.DL_HISTORY_DIR = os.path.join(self.EXP_DIR, 'performance_history')
        self.METRICS_DIR = os.path.join(self.EXP_DIR,'metrics_ml')
        TIMES_DIR = os.path.join(self.EXP_DIR, 'train_pred_times')
        self.FOLD_TIMES_DIR = os.path.join(TIMES_DIR, 'times_kfold')
        self.MEAN_TIMES_DIR = os.path.join(TIMES_DIR, 'times_mean')


        
        # Check if the directory exists
        if os.path.exists(self.EXP_DIR):
            # Remove the existing directory and its contents
            shutil.rmtree(self.EXP_DIR)
        # Create the new directory
        os.makedirs(self.EXP_DIR)


        if self.save_models:
            os.makedirs(self.SAVED_MODELS_DIR)
        os.makedirs(self.DL_HISTORY_DIR)
        os.makedirs(self.METRICS_DIR)
        os.makedirs(TIMES_DIR)
        os.makedirs(self.FOLD_TIMES_DIR)
        os.makedirs(self.MEAN_TIMES_DIR)
        
        print(f"\nEXPERIMENT IDENTIFIER {self.expID}. In /outputs/{self.expID}")
        
    
    #######################################################

    ### For log.txt file

    def setup_logging(self, log_file_path):
        """Start logging all terminal outputs to the specified file."""
        self.original_stdout = sys.stdout  # store the original stdout
        # redirect stdout to DualOutputStream to write both to file and terminal
        sys.stdout = DualOutputStream(log_file_path, self.original_stdout)
        print("\nTerminal output for models_evaluation.py")
        return sys.stdout

    def stop_logging(self):
        """Stops logging and reverts output back to the terminal."""
        if hasattr(self, 'original_stdout') and self.original_stdout:
            sys.stdout.close()
            sys.stdout = self.original_stdout
    
    ########################################################################################################
    ########################################################################################################
    ### ---------------------------------------------------------------
    ### Functions for configuring parameters depending on problem type
    ### ---------------------------------------------------------------

    def check_target_type(self, y):
        unique_values = np.unique(y)
        is_continuous = False
        is_multiclass = False
        
        if len(unique_values) > self.regression_threshold: 
            is_continuous = True

        # Check if any unique value is a non-integer float.
        for value in unique_values:
            if isinstance(value, float) and not value.is_integer(): 
                is_continuous = True
                break

        # If not continuous, it's binary or multiclass.
        if not is_continuous:
            is_multiclass = len(unique_values) > 2  # Multiclass if more than 2 unique values.
                                                    # and less than regression_threshold

        num_classes = math.inf if is_continuous else len(unique_values)

        return is_continuous, is_multiclass, num_classes

    def set_loss_function(self):
        if self.num_classes == math.inf: # Regression
            self.loss = 'mean_squared_error'
        else:  # Classification
            if self.num_classes > 2:
                self.loss = 'categorical_crossentropy'
            else:
                self.loss = 'binary_crossentropy'

        print(f"For {self.num_classes} classes, the loss function is {self.loss}")
       
    ########################################################################################################
    ### -----------
    ### Main Loop
    ###------------

    """
    Function that loops throguh all models belonging to a module 
    A module: 'Machine Learning', 'Deep Learning' or 'Transformers'
    For a dataset, each model will either, run once, or run through Kfold.
    """
    def process_all_models_in_module(self, metric_module, module_key, module, dataset_name, dataset_file, language):

        # Load dataset
        X, y = module.load_data(dataset_file)  

        # Determine defining parametres
        num_features = 100  # placeholder
        if module_key != 'Transformers':
            num_features = module.load_glove_model(language)
        
        self.is_continuous, self.is_multiclass, self.num_classes = self.check_target_type(y)

        # Retrieve all models (belonging to each module)
        all_models = AllModels(num_features, self.num_classes, self.is_continuous)
        models = all_models.get_models(module_key)
        print(f"\nMODELS in {module_key}")
        print(models)
        print()

        # Get appropriate evaluation metric
        self.eval_metric = self.metric_criterion['continuous'] if self.is_continuous else self.metric_criterion['multiclass'] if self.is_multiclass else self.metric_criterion['binary']

        print(f"Dataset: {dataset_name}")
        print(f"Module: {module_key}")
        print(f"number of features: {num_features}")
        print(f"number of classes: {self.num_classes}")
        print(f"is_continuous:{self.is_continuous}")
        print(f"is_multiclass: {self.is_multiclass}")
        
        
        # Loop through all models
        for model_name, model, hyperparams in models:
            print(f"\n------------------------------------")
            print(f"Model: {model_name}")
            
            # Additional step for TF module
            if module_key == 'Transformers':
                module.set_model(model_name)

            # Reset current_best_score at the beginning of each Kfold for each model
            self.current_best_score = 0 if not self.is_continuous else math.inf 
            
            model_tuple = (model_name, model, hyperparams)

            if self.k == 1:
                print(f"\nKFOLD INACTIVE:")
                self.process_one_fold(module_key, module, metric_module, X, y, model_tuple, dataset_name)
            else: # K-Folds cross validator for each model
                kf = self.create_folds()
                print(f"\nKFOLD ACTIVE:")
                for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
                    print(f"\nk={fold}")
                    print(f"Model: {model_name}")
                    self.process_fold(fold, module_key, module, metric_module, train_index, test_index, X, y, model_tuple, dataset_name)
                
            print(f"\nIn dataset: {dataset_name}")
            print(f"Finished model: {model_name}")
            print(f"Model with current highest metric score:")
            print(f"{self.eval_metric}: {self.current_best_score}")

            metric_module.compute_metric_statistics(module_key, dataset_name, model_name)
          
        if self.save_history_performance:
            if module_key != 'Machine Learning':
                self.compute_history_mean(dataset_name)
                self.save_history_to_csv(dataset_name, module_key)

        metric_module.save_results(module_key)

    ########################################################################################################
    ### Per 
    # K = 1:
    def process_one_fold(self, module_key, module, metric_module, X, y, model_tuple, dataset_name):
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.set_loss_function()

        # Hyperparameter tuning before training
        if self.tune:
            print(f"\nHYPERPARAMETER OPTIMIZATION ACTIVE")
            model_tuple = self.perform_tuning(module_key, module, model_tuple, X_train, y_train)
        else:
            print(f"\nHYPERPARAMETER OPTIMIZATION INACTIVE")

        # Training, prediction and evaluation
        fold = 0
        trained_model, fold_results = self.train_predict_evaluate(fold, module_key, module, model_tuple, metric_module, dataset_name, X_train, y_train, X_test, y_test)

        # Through folds compare and save best model 
        self.update_saved_best_model(trained_model, dataset_name, model_tuple, fold_results, module_key, module)

    ########################################################################################################
    # K > 1:
    def create_folds(self):
        if self.is_continuous:
            kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
            print("KFold")
        else:
            kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
            print("StratifiedKFold")

        return kf

    def split_data(self, X, y, train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

    def process_fold(self, fold, module_key, module, metric_module, train_index, test_index, X, y, model_tuple, dataset_name):
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, train_index, test_index)
        self.set_loss_function()

        # Hyperparameter tuning before training
        if self.tune:
            print(f"\nHYPERPARAMETER OPTIMIZATION ACTIVE")
            model_tuple = self.perform_tuning(module_key, module, model_tuple, X_train, y_train)
        else:
            print(f"\nHYPERPARAMETER OPTIMIZATION INACTIVE")

        # Training, prediction and evaluation
        trained_model, fold_results = self.train_predict_evaluate(fold, module_key, module, model_tuple, metric_module, dataset_name, X_train, y_train, X_test, y_test)

        # Through folds compare and save best model 
        self.update_saved_best_model(trained_model, dataset_name, model_tuple, fold_results, module_key, module)

    ########################################################################################################

    def perform_tuning(self, module_key, module, model_tuple, X_train, y_train):
        
        model_name, model_func, params = model_tuple 
        best_params = module.hyperparameter_tuning(model_tuple, X_train, y_train, self.is_continuous, self.loss, self.num_classes, self.tune_method)  # tuning

        if best_params:
                if module_key == 'Machine Learning':
                    print(f"\nBest parameters for {model_name}: {best_params}")
                elif module_key == 'Deep Learning':
                    print(f"\nBest parameters for {model_name}: {best_params.values}")
        else:
            print(f"\n{model_name} has no hyperparameters to tune or tuning did not return any improved parameters.")

        # Update the model with the best parameters
        if module_key == 'Machine Learning':
            model_func.set_params(**best_params)
        elif module_key == 'Deep Learning':
            model_func = module.build_model_with_best_params(best_params)
        elif module_key == 'Transformers':
            model_func = module.update_model_with_best_params(model_func, best_params, self.loss)

        model_tuple = (model_name, model_func, best_params)
        return model_tuple
    
    ########################################################################################################
    ### Train, Predict and evaluate
    # A given dataset, a given model, one fold.
    def train_predict_evaluate(self, fold, module_key, module, model_tuple, metric_module, dataset_name, X_train, y_train, X_test, y_test):
        model_name = model_tuple[0] 
        
        if module_key == 'Machine Learning':
            trained_model, train_time = module.train(X_train, y_train, X_test, y_test, model_tuple, self.num_classes, self.loss)
            print("\nTraining completed")
        else:
            trained_model, train_time, history = module.train(X_train, y_train, X_test, y_test, model_tuple, self.num_classes, self.loss)
            print("\nTraining completed")
            # Extract the history
            train_loss = history.history['loss']
            train_accuracy = history.history['accuracy']
            val_loss = history.history['val_loss']
            val_accuracy = history.history['val_accuracy']

            # Optionally, you can print the loss and accuracy
            #print(f"Train Loss: {train_loss}")
            #print(f"Train Accuracy: {train_accuracy}")
            print(f"Train Loss: {[f'{loss:.4f}' for loss in train_loss]}")
            print(f"Train Accuracy: {[f'{acc:.4f}' for acc in train_accuracy]}")
            print(f"Validation Loss: {val_loss}")
            print(f"Validation Accuracy: {val_accuracy}")

            # Saving history
            # History is the performance across epochs when training a Deep Learning model
            self.history_folds[dataset_name][model_name][fold]['train_loss'] = train_loss
            self.history_folds[dataset_name][model_name][fold]['train_accuracy'] = train_accuracy
            self.history_folds[dataset_name][model_name][fold]['val_loss'] = val_loss
            self.history_folds[dataset_name][model_name][fold]['val_accuracy'] = val_accuracy

            #val_loss, val_accuracy = trained_model.evaluate(X_val_processed, y_val_processed, verbose=0)
            #print(f"Validation Loss: {val_loss}")
            #print(f"Validation Accuracy: {val_accuracy}")

        predictions, predict_time = module.predict(trained_model, X_test, self.num_classes)
        print("Prediction completed")
        
        """
        print("\nCOMPARISON: ground truth - prediction")
        print(f"\ny_test:\n{y_test}")
        predictions_series = pd.Series(predictions, index=y_test.index)
        print(f"\npredictions:\n{predictions_series}")
        
        # side-by-side:
        if module_key=='Transformers':
            print("No print comparison for TF.")
            # y_test is a numpy array , use length as index
            #predictions_series = pd.Series(predictions, name='predictions')
            #y_test_series = pd.Series(y_test, name='y_test')
            #comparison_df = pd.concat([y_test_series, predictions_series], axis=1)
            #print("\nCOMPARISON: ground truth vs. prediction")
            #print(comparison_df)
        else:
            predictions_series = pd.Series(predictions, index=y_test.index, name='predictions')
            comparison_df = pd.concat([y_test, predictions_series], axis=1)
            print("\nCOMPARISON: ground truth vs. prediction")
            print(comparison_df)
        """
     
        fold_results = metric_module.calculate_metrics(y_test, predictions, self.is_continuous, self.is_multiclass, module_key)
        metric_module.update_results(module_key, dataset_name, model_name, fold_results)
        print("Evaluation completed")

        self.save_train_predict_times(dataset_name, model_name, fold, train_time, predict_time)

        return trained_model, fold_results
    
    ###################################################################################################
    ###-----------------------------------------------------------------------------------------------
    ### Functions for saving History of Performance when Training Deep Learning Models across epochs
    ### Such as: Train Accuracy, Train Loss, Validation Accuracy, Validation Loss.
    ###-----------------------------------------------------------------------------------------------
    

    def compute_history_mean(self, dataset_name):
    
        # Nested Dictionary:
        # self.history_folds[dataset_name][model_name][fold]['train_loss']
        # Example of 'train loss', 'accuracy loss':
                
        #Train Loss: [0.6725, 0.6029, 0.5609, 0.5446, 0.5379, 0.5252, 0.5215, 0.5104, 0.5076, 0.5059, 0.5004, 0.4905, 0.4820, 0.4737, 0.4758]
        #Train Accuracy: [0.5823, 0.6857, 0.7202, 0.7298, 0.7391, 0.7439, 0.7485, 0.7542, 0.7573, 0.7621, 0.7647, 0.7690, 0.7755, 0.7780, 0.7781]
        
        models_dict = self.history_folds[dataset_name]
        for model_name, _ in models_dict.items():

            # Initialize the sum lists with zeros. 
            sum_train_loss = [0] * len(self.history_folds[dataset_name][model_name][0]['train_loss'])
            sum_train_accuracy = [0] * len(self.history_folds[dataset_name][model_name][0]['train_accuracy'])
            sum_val_loss = [0] * len(self.history_folds[dataset_name][model_name][0]['val_loss'])
            sum_val_accuracy = [0] * len(self.history_folds[dataset_name][model_name][0]['val_accuracy'])


            # Per dataset, per model, mean of all folds.
            print(f"Computing history mean...")
            for fold in range(self.k):
                print(f"Retrieving data in dataset: {dataset_name}, model: {model_name}, fold: {fold}")

                fold_train_loss = self.history_folds[dataset_name][model_name][fold]['train_loss']
                fold_train_accuracy = self.history_folds[dataset_name][model_name][fold]['train_accuracy']
                fold_val_loss = self.history_folds[dataset_name][model_name][fold]['val_loss']
                fold_val_accuracy = self.history_folds[dataset_name][model_name][fold]['val_accuracy']


                # Sum the elements from the current fold into the cumulative sum lists
                sum_train_loss = [sum(x) for x in zip(sum_train_loss, fold_train_loss)]
                sum_train_accuracy = [sum(x) for x in zip(sum_train_accuracy, fold_train_accuracy)]
                sum_val_loss = [sum(x) for x in zip(sum_val_loss, fold_val_loss)]
                sum_val_accuracy = [sum(x) for x in zip(sum_val_accuracy, fold_val_accuracy)]
                
            # Calculate the mean of train_loss and train_accuracy by dividing each element by self.k
            train_loss_mean = [x / self.k for x in sum_train_loss]
            train_accuracy_mean = [x / self.k for x in sum_train_accuracy]
            val_loss_mean = [x / self.k for x in sum_val_loss]
            val_accuracy_mean = [x / self.k for x in sum_val_accuracy]

            # Calculate the mean for each metric and store it in the new dictionary
            self.history_mean[dataset_name][model_name] = {
                'train_loss_mean': train_loss_mean,
                'train_accuracy_mean': train_accuracy_mean,
                'val_loss_mean': val_loss_mean,
                'val_accuracy_mean': val_accuracy_mean
            }


    def save_history_to_csv(self, dataset_name):

        # Assuming self.history_mean is already populated with the mean values as shown before
        metrics = ['train_loss_mean', 'train_accuracy_mean', 'val_loss_mean', 'val_accuracy_mean']
        #metrics = ['train_loss_mean', 'train_accuracy_mean']

        for metric in metrics:
            # Create a dictionary that will be used to build the DataFrame
            data_for_csv = {}
            for dataset_name, models in self.history_mean.items():
                data_for_csv[dataset_name] = []
                for model_name in models:
                    # Append the metric for each model under the appropriate dataset
                    data_for_csv[dataset_name].append(models[model_name][metric])

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(data_for_csv)

            # Optionally, if you want to include the model names as the first column:
            df.insert(0, 'Model', list(self.history_mean[next(iter(self.history_mean))].keys()))

            # Directory
            history_dir = os.path.join(self.DL_HISTORY_DIR, self.expID)
            os.makedirs(history_dir, exist_ok=True)
            file_path = os.path.join(history_dir, f'{dataset_name}_{metric}.csv')

            # Save to a CSV file
            df.to_csv(file_path, index=False)
       
            
    #######################################################################################
    ### Compute train and predict times
    """
    If KFold was applied, the mean train time and predcit time across folds will be computed
    Otherwise, mean time is directly the time of the 1 execution.
    """
    def compute_mean_time(self):
        train_times = defaultdict(lambda: defaultdict(float))
        predict_times = defaultdict(lambda: defaultdict(float))
        
        # Loop through each CSV file in the times/kfold/ directory
        for file_name in os.listdir(self.FOLD_TIMES_DIR):

            # Assuming the structure is always {dataset_name}_{model_name}
            #dataset_name, model_name = file_name.replace("_times.csv", "").split("_")
            # if dataset_name containts "_"
            # Remove the '_times.csv' part
            file_name_without_suffix = file_name[:-len("_times.csv")]

            # Split from the right (starting from the end) at the first underscore
            dataset_name, model_name = file_name_without_suffix.rsplit("_", 1)

            
            # Read each CSV file
            file_path = os.path.join(self.FOLD_TIMES_DIR, file_name)
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                train_time_list = []
                predict_time_list = []
                
                for row in reader:
                    fold, train_time, predict_time = row
                    train_time_list.append(float(train_time))
                    predict_time_list.append(float(predict_time))
                
                # Compute the mean times
                train_times[dataset_name][model_name] = np.mean(train_time_list)
                predict_times[dataset_name][model_name] = np.mean(predict_time_list)
        
        # Convert the nested dictionaries to DataFrames for easier CSV writing
        train_times_df = pd.DataFrame(train_times).T
        predict_times_df = pd.DataFrame(predict_times).T
        
        # Save to CSV
        mean_dir = self.MEAN_TIMES_DIR
        train_times_df.to_csv(os.path.join(mean_dir, 'train_times_mean.csv'))
        predict_times_df.to_csv(os.path.join(mean_dir, 'predict_times_mean.csv'))

        return

    # Train and Predict times:
    def save_train_predict_times(self, dataset_name, model_name, fold, train_time, predict_time):

        # Saving times to a CSV file within kfold directory
        file_path = os.path.join(self.FOLD_TIMES_DIR, f'{dataset_name}_{model_name}_times.csv')
    
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([fold, train_time, predict_time])
        return

    ########################################################################################################
    ###------------------
    ### Save Model File
    ###------------------

    def update_saved_best_model(self, trained_model, dataset_name, model_tuple, fold_results, module_key, module):
        model_name, _, _ = model_tuple
        print(f"\nThe best model for dataset '{dataset_name}' will be selected")
        print(f"using the metric '{self.eval_metric}' as criterion")

        # Retrieve actual metric for current model to compare with previous best model metric
        eval_metric_value = fold_results.get(self.eval_metric)                                               

        if self.is_continuous:
            if self.current_best_score > eval_metric_value: # For continuous, lower is better (RMSE)
                self.current_best_score = eval_metric_value # Update current best score
                print(f"Current best metric score: {self.current_best_score}")
                if self.save_models:
                    module.save_model(trained_model, model_name, dataset_name, self.SAVED_MODELS_DIR)  # pass the experiment directory
                    print(f"Saving model {model_name}...")
                
        else:
            if self.current_best_score < eval_metric_value: # For classification, higher is better (accuracy)
                self.current_best_score = eval_metric_value # Update current best score
                print(f"Current best metric score: {self.current_best_score}")
                if self.save_models:
                    module.save_model(trained_model, model_name, dataset_name, self.SAVED_MODELS_DIR)  # pass the experiment directory
                    print(f"Saving model {model_name}...")
                
    ########################################################################################################

    def main(self):
        
        with open(CONFIG_ML, 'r') as f:
            config = json.load(f)

        # Load dataset info from json file and append the dataset csv absolute path 
        datasets_info = [tuple(item) for item in config['datasets_processed']]
        datasets = [(name, os.path.join(DATA_PROCESSED, filepath), language) for name, filepath, language, _ in datasets_info]

        modules_dict = {
            'Machine Learning': MachineLearningModule(),
            'Deep Learning': DeepLearningModule(),
            'Transformers': TransformersModule()
        }
        # This will get active modules as a dictionary filtered by config
        active_modules_dict = {name: instance for name, instance in modules_dict.items() if config['modules'][name]['active']}
        # Convert the dictionary to a list of tuples
        active_modules = list(active_modules_dict.items())

        """
        This is the format of active_modules:
        active_modules = [  ('Machine Learning', MachineLearningModule()),
                            ('Deep Learning', DeepLearningModule()),
                            ('Transformers', TransformersModule())]
        """

        # The directory path passed as argument is where the final metrics will be stored.
        metric_module = MetricsCalculator(metrics_dir=self.METRICS_DIR)  
        
        # Main loop
        for (dataset_name, dataset_file, language) in datasets: 
            for module_key, module in active_modules: # modules: ML, DL, TF
                self.process_all_models_in_module(metric_module, module_key, module, dataset_name, dataset_file, language)
        
        # After all models-datasets have been processed, merge the results
        # Optional. This is incase we want the models from modules (ML, DL, TF) mixed
        # metric_module.merge_all_results()

        # compute mean train and predict times
        self.compute_mean_time()

# Main Execution
if __name__ == "__main__":
    evaluator = ModelsEvaluation()
    evaluator.main()
    if evaluator.save_log:
        evaluator.stop_logging()

