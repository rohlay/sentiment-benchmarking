"""
Author: Rohan Laycock

"""

import os
import csv
import glob
import json
import time
import pandas as pd
import numpy as np
import sys
from afinn import Afinn
import nltk
nltk.download('wordnet')  
nltk.download('sentiwordnet')
from nltk.corpus import wordnet, sentiwordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from pattern.en import sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
# Importing config module

from sentiment_analysis_project.scripts.config_dir import GENERAL_INQUIRER_FILE, MPQA_FILE, OPINIONFINDER_FILE
from sentiment_analysis_project.scripts.config_dir import DATA_PROCESSED
from sentiment_analysis_project.scripts.config_dir import CONFIG_LEXICON
from sentiment_analysis_project.scripts.config_dir import OUTPUTS_DIR

from sentiment_analysis_project.input_data.lexicon_files.SenticNet.senticnet import senticnet # import dictionary (not module)

compute_metric_correctedness = True # Correctedness = 1 - RMSE


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


class LexiconEvaluation:
    def __init__(self):
        self.ID = None
        self.EXP_DIR = None
        self.ORIG_METRICS_DIR = None
        self.NORM_METRICS_DIR = None
        self.create_new_exp_dirs()

        with open(CONFIG_LEXICON, 'r') as f:
                self.config = json.load(f)
        self.save_log = self.config["save_log"]
        if self.save_log:
            log_file_path = os.path.join(self.EXP_DIR, f"{self.ID}.txt")
            log_file = self.setup_logging(log_file_path) # Setup the log file for terminal outputs
            # Saving a copy of the config_lex.json inside the ID directory
            with open(CONFIG_LEXICON, 'r') as source_file:
                data = source_file.read()
                with open(os.path.join(self.EXP_DIR, 'config_lex.json'), 'w') as target_file:
                    target_file.write(data)
        

    def create_new_exp_dirs(self):
            
            os.makedirs(OUTPUTS_DIR, exist_ok=True) # Create /output_results, if it doesn't exist.            
            today = time.strftime("%Y_%m_%d") # Get today's date in the format YYYYMMDD

            # Get all directories that start with today's date
            dirs_today = sorted([os.path.basename(path) for path in glob.glob(f"{OUTPUTS_DIR}/{today}_*")])

            # Determine new number based on existing directories
            if dirs_today:
                last_num_today = int(dirs_today[-1].split('_')[-1])
                new_num_today = last_num_today + 1
            else:
                new_num_today = 1

            self.ID = f"{today}_{new_num_today}"
            self.EXP_DIR = f'{OUTPUTS_DIR}/{today}_{new_num_today}'
            os.makedirs(self.EXP_DIR, exist_ok=True)

            self.ORIG_METRICS_DIR = os.path.join(self.EXP_DIR, "original")
            self.NORM_METRICS_DIR = os.path.join(self.EXP_DIR, "normalized")
            os.makedirs(self.ORIG_METRICS_DIR, exist_ok=True)
            os.makedirs(self.NORM_METRICS_DIR, exist_ok=True)

            print(f"\nEXPERIMENT WITH IDENTIFIER {today}_{new_num_today}")
            

    def load_data(self, dataset):
        df = pd.read_csv(dataset)
        X = df['text']
        y = df['sentiment']

        return X, y
        

    def calculate_continuous_metrics_and_normalized(self, y_true, y_pred, data_range): # y_max, y_min
        # If y_true and y_pred have a range differrent from (-1,1), then calculate normalized metrics (0,1)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)

        # norm
        y_min, y_max = data_range
        y_error_max = abs(y_max - y_min)
        print(f"\nNormalizing metrics...")
        print(f"Dataset range: y_min = {y_min}, y_max = {y_max}")
        print(f"Maximum error = {y_error_max}")
        print(f"The maximum error is the division factor to normalize mse and mae")

        metrics = {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'r2_score': r2,
            'explained_variance_score': evs}
        
        mse_norm = mse/(y_error_max) ** 2
        mae_norm = mae/(y_error_max)
        rmse_norm = rmse/(y_error_max)
        correctedness = 1 - rmse_norm,

        """
        metrics_norm = {
            'mean_squared_error': mse_norm,
            'mean_absolute_error': mae_norm,
            'root_mean_squared_error': rmse_norm,
            'correctedness': correctedness,
            'r2_score': r2,
            'explained_variance_score': evs}
        """

        metrics_norm = {
            'mean_squared_error': mse_norm,
            'mean_absolute_error': mae_norm,
            'root_mean_squared_error': rmse_norm,
            'correctedness': correctedness,
        }


        return metrics, metrics_norm

    def normalize_score(self, score, text):
        num_words = len(text.split())
        return score / num_words if num_words else 0

    def apply_lexicon_method(self, method, X):
        if method == 'AFINN':
            #predictions = X.apply(afinn.score)
            predictions = X.apply(lambda x: self.normalize_score(afinn.score(x), x))
            return predictions

        elif method == 'General Inquirer':
            # load the General Inquirer lexicon file (using read_excel for xls file)
            gi_file = GENERAL_INQUIRER_FILE
            gi_df = pd.read_excel(gi_file)

            # create a dictionary to map words to sentiment scores
            gi_dict = {}
            for index, row in gi_df.iterrows():
                word = row['Entry']
                if pd.notnull(row['Positiv']):
                    gi_dict[word] = 1 # assign a positive score
                elif pd.notnull(row['Negativ']):
                    gi_dict[word] = -1 # assign a negative score
                else:
                    gi_dict[word] = 0 # assign a neutral score if neither positive nor negative

            # apply the dictionary to your dataset and return the scores
            #predictions = X.apply(lambda x: sum([gi_dict.get(word, 0) for word in x.split()]))
            predictions = X.apply(lambda x: self.normalize_score(sum([gi_dict.get(word, 0) for word in x.split()]), x))
            return predictions

        elif method == 'MPQA':
            # load the MPQA lexicon file and map the scores to the dataset
            mpqa_file = MPQA_FILE
            #mpqa_file = 'mpqa.tff'
            mpqa_dict = {}
            with open(mpqa_file, 'r') as f:
                for line in f:
                    if line.startswith('type='):
                        parts = line.strip().split()
                        word = parts[2].split('=')[1]
                        polarity = parts[-1].split('=')[1]
                        if polarity == 'positive':
                            mpqa_dict[word] = 1
                        elif polarity == 'negative':
                            mpqa_dict[word] = -1
                        else:
                            mpqa_dict[word] = 0

            predictions = X.apply(lambda x: self.normalize_score(sum([mpqa_dict.get(word, 0) for word in x.split()]), x))
            #predictions = X.apply(lambda x: sum([mpqa_dict.get(word, 0) for word in x.split()]))
            return predictions

        elif method == 'OpinionFinder':
            #opinionfinder_file = 'subjcluesSentenceClassifiersOpinionFinderJune06.tff'
            opinionfinder_file = OPINIONFINDER_FILE
            opinionfinder_dict = {}
            with open(opinionfinder_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    else:
                        parts = line.strip().split()
                        word = parts[2].split('=')[1]
                        polarity = parts[-1].split('=')[1]
                        if polarity == 'strongpos':
                            opinionfinder_dict[word] = 1
                        elif polarity == 'weakpos':
                            opinionfinder_dict[word] = 0.5
                        elif polarity == 'strongneg':
                            opinionfinder_dict[word] = -1
                        elif polarity == 'weakneg':
                            opinionfinder_dict[word] = -0.5
                        else:
                            opinionfinder_dict[word] = 0

            #predictions = X.apply(lambda x: sum([opinionfinder_dict.get(word, 0) for word in x.split()]))
            predictions = X.apply(lambda x: self.normalize_score(sum([opinionfinder_dict.get(word, 0) for word in x.split()]), x))
            return predictions

        elif method == 'Pattern':
            #predictions = X.apply(lambda x: sum([sent.polarity for sent in TextBlob(x).sentences]))
            predictions = X.apply(lambda x: sentiment(x)[0])
            return predictions
        
        elif method == 'SenticNet':
            # create the SenticNet dictionary and map the polarity values
            senticnet_dict = {word: values[7] for word, values in senticnet.items()}  # Index 7 corresponds to 'polarity_value'
            # The lambda function takes each text in X, splits it into individual words using the split() method, 
            # and then performs a sum of the polarity values (sentiment scores) for each word according to the SenticNet dictionary.
            
            #predictions =  X.apply(lambda x: sum([senticnet_dict.get(word, 0) for word in x.split()]))
            predictions = X.apply(lambda x: self.normalize_score(sum([senticnet_dict.get(word, 0) for word in x.split()]), x))

            return predictions

        elif method == 'TextBlob':
            predictions = X.apply(lambda x: TextBlob(x).sentiment.polarity)
            return predictions

        elif method == 'VADER':
            predictions = X.apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])
            return predictions

        elif method == 'WordNet-Affect':
        #elif method == 'SentiWordNet': # TODO
            # SentiWordNEt ?
            # map the WordNet-Affect scores to the dataset
            def wordnet_score(word):
                synsets = wordnet.synsets(word)
                if not synsets:
                    return 0
                else:
                    synset = synsets[0] # Using the first synset
                    swn_synset = sentiwordnet.senti_synset(synset.name())
                    return swn_synset.pos_score() - swn_synset.neg_score()

            def calculate_sentiment(text):
                wordnet_lemmatizer = WordNetLemmatizer()
                tokens = word_tokenize(text)
                sentiment_score = sum([wordnet_score(wordnet_lemmatizer.lemmatize(word)) for word in tokens])
                return sentiment_score

            #predictions = X.apply(calculate_sentiment)
            predictions = X.apply(lambda x: self.normalize_score(calculate_sentiment(x), x))
            return predictions

        else:
            raise ValueError('Invalid lexicon method')

    def map_prediction(self, prediction, method):
        new_low, new_high = -1, 1

        # Defining method-specific ranges
        method_ranges = {
            'AFINN': (-5, 5),
            'General Inquirer': (-1, 1),
            'MPQA': (-1, 1),
            'OpinionFinder': (-1, 1), 
            'Pattern': (-1, 1),
            'SenticNet': (-1, 1),
            'TextBlob': (-1, 1),
            'VADER': (-1, 1),
            'WordNet-Affect': (-1, 1) # "MISTAKE" Output is (0, 1). But im doing positive [0,1] - negative[0,1] = [-1, +1]. "WordNet" NOT "WordNet-Affect"
        }

        method_range = method_ranges.get(method)
        if method_range is None:
            raise ValueError(f"Unknown method: {method}")

        # Mapping the prediction from its original range to the specified range
        old_low, old_high = method_range
        mapped_prediction = new_low + (prediction - old_low) * (new_high - new_low) / (old_high - old_low)

        return mapped_prediction

    def map_ground_truth_values(self, y_true, data_range):
        # linear transformation, min-max scaling, or rescaling
        
        old_low, old_high = data_range # Unpack the mapping_values for the dataset
        new_low, new_high = -1, 1 # Mapping the true values from their original range to the range of -1 to 1

        m = (new_high - new_low) / (old_high - old_low)
        mapped_ground_truth = new_low + m * (y_true - old_low)

        self.print_mappings(y_true, mapped_ground_truth)

        return mapped_ground_truth
    

    ######################################################
    # Print results

    def print_mappings(self, y_true, mapped_values):
        # Create a set to store unique Original Value (y_true) and their corresponding Mapped Value
        unique_mapping = set()

        # Create a list to store the mappings in ascending order of the Original Value (y_true)
        ordered_mappings = []

        # Iterate through the mappings and add them to the ordered_mappings list in the desired order
        for original_val, mapped_val in zip(y_true, mapped_values):
            if original_val not in unique_mapping:
                ordered_mappings.append((original_val, mapped_val))
                unique_mapping.add(original_val)

        # Sort the mappings based on the Original Value (y_true) in ascending order
        ordered_mappings.sort(key=lambda x: x[0])

        # Print the ordered mappings
        self.print_ordered_mappings(ordered_mappings)

    def print_ordered_mappings(self, ordered_mappings):
        print(f"\nOriginal Value (y_true) | Mapped Value")
        print("------------------------|--------------")

        if len(ordered_mappings) > 20:
            to_print = ordered_mappings[:4] + ordered_mappings[-4:] # reduce to show only first and last 4 elements
        else:
            to_print = ordered_mappings # show all elements

        for original_val, mapped_val in to_print:
            print(f"{original_val:<23} | {mapped_val:.4f}")

    ######################################################
    # Save csv

    def save_results_to_csv(self, results):
        for result_type, datasets in results.items():
            for dataset_name, models in datasets.items():
                if not models: # Check if models dictionary is empty
                    print(f"No metrics found for {dataset_name}. Skipping...")
                    continue
                
                # Extracting metric names from the first model
                metrics = list(next(iter(models.values())).keys())
                
                for metric in metrics:
                    if result_type == "original":
                        file_path = os.path.join(self.ORIG_METRICS_DIR, f"{metric}.csv")
                    else:
                        file_path = os.path.join(self.NORM_METRICS_DIR, f"{metric}.csv")

                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    mode = 'a' if os.path.exists(file_path) else 'w'
                    with open(file_path, mode, newline='') as file:
                        writer = csv.writer(file)
                        if mode == 'w':
                            writer.writerow(['Dataset Name'] + list(models.keys()))
                        
                        # Writing a row with dataset name and metric values for all models
                        writer.writerow([dataset_name] + [model_metrics.get(metric, '') for model_metrics in models.values()])
    
    ######################################################
    # Log file:
    def setup_logging(self, log_file_path):
        """Start logging all terminal outputs to the specified file."""
        global original_stdout 
        original_stdout = sys.stdout  # store the original stdout
        # redirect stdout to DualOutputStream to write both to file and terminal
        sys.stdout = DualOutputStream(log_file_path, original_stdout)
        print("\nTerminal output for models_evaluation.py")
        return sys.stdout

    def stop_logging(self, ):
        """Stops logging and reverts output back to the terminal."""
        if original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
    ######################################################
    def main(self):
        
        datasets_info = [tuple(item) for item in self.config['datasets_processed']]
        datasets = [(name, os.path.join(DATA_PROCESSED, filepath), language, data_range) for name, filepath, language, data_range in datasets_info]

        # Initialize a dictionary to store results outside the loop
        results_dict = {
            "original": {dataset_name: {} for dataset_name, _, _, _ in datasets},
            "normalized": {dataset_name: {} for dataset_name, _, _, _ in datasets},
        }
        
        ### MAIN LOOP:
        for (dataset_name, dataset_file, language, data_range) in datasets: 

            print(f"Dataset range {data_range}")
            if data_range != [-1,1]:
                print(f"Detected range different to (-1,1), will need mappping...")
            
            X, y = self.load_data(dataset_file)
            models = [model for model, value in self.config.get("models", {}).items() if value]

            for model in models:
                # Ground truth
                if data_range != [-1,1]:
                    y_true_mapped = self.map_ground_truth_values(y, data_range)
                else:
                    y_true_mapped = y

                # Predictions
                predictions = self.apply_lexicon_method(model, X) 
                if model == 'AFINN':    # Models that need mapping (output is different from (-1,1))
                    predictions_mapped = self.map_prediction(predictions, model)
                else:
                    predictions_mapped = predictions

                print(f"\nModel {model}")
                print(f"Ground Truth, y_true:\n{y_true_mapped}")
                print(f"Predictions, y:\n{predictions_mapped}")

                metric_results, metric_results_normalized = self.calculate_continuous_metrics_and_normalized(y_true_mapped, predictions_mapped, [-1,1])
    

                # Some checks:
                # - Exact macthes:
                matches = sum(1 for a, b in zip(y_true_mapped, predictions_mapped) if a == b)
                print(f"{matches} out of {len(y_true_mapped)} are exact matches.")

                # - Outliers:
                print("y_true outliers:", [val for val in y_true_mapped if val < -1 or val > 1])
                print("y_pred outliers:", [val for val in predictions_mapped if val < -1 or val > 1])
                outlier_indices = [index for index, val in enumerate(y_true_mapped) if val < -1 or val > 1]
                print("Indices of y_true outliers:", outlier_indices)

                # Print Metrics:
                print(f"\nResults for {dataset_name} using {model}: {metric_results}")
                print(f"Results Normalized for {dataset_name} using {model}: {metric_results_normalized}")

                # Store original and normalized results in the dictionary
                results_dict["original"][dataset_name][model] = metric_results
                results_dict["normalized"][dataset_name][model] = metric_results_normalized

        # Save the results to CSV
        self.save_results_to_csv(results_dict)

# Main Execution
if __name__ == "__main__":

    # Initialize the lexicon-based models
    print("Inititializing AFFIN")
    afinn = Afinn()
    print("Inititializing WordNet Lemmatizer..") 
    wordnet_lemmatizer = WordNetLemmatizer()
    print("Inititializing VADER Intensity Analyzer")
    vader_analyzer = SentimentIntensityAnalyzer()

    evaluator = LexiconEvaluation()
    evaluator.main()

    if evaluator.save_log:
        evaluator.stop_logging()