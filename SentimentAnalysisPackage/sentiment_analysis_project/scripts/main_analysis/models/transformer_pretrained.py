import os
import json
import time
import numpy as np
import pandas as pd
import math
import glob
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

from sentiment_analysis_project.scripts.config_dir import CONFIG_ML

class TransformersModule:
    def __init__(self):
        self.tokenizer = None
        self.transformer_model = None
        self.max_length = 128  # maximum sequence length for your inputs
        self.label_encoder = LabelEncoder()

        with open(CONFIG_ML, 'r') as f:
            config = json.load(f)
        transformers_config = config['modules']['Transformers']
        self.epochs = transformers_config['epochs']
        self.batch_size = transformers_config['batch_size']

    def load_data(self, dataset_file):
        df = pd.read_csv(dataset_file)
        X = df['text'].values
        y = df['sentiment'].values

        # Encode labels into integers
        self.label_encoder.fit_transform(y) # and returns encoded labels (not stored)
        #self.label_encoder.fit(y)

        return X, y

    def tf_model_path(self, model_name):
        models_path = {   
            'bert-base-uncased': 'bert-base-uncased',
            'bert-base-multilingual-cased': 'bert-base-multilingual-cased',
            'distilbert-base-uncased': 'distilbert-base-uncased',
            'roberta-base': 'roberta-base', 
            'robertuito-sentiment-analysis': 'pysentimiento/robertuito-sentiment-analysis',
            'electra-base-discriminator': 'google/electra-base-discriminator', 
            'albert-base-v2': 'albert-base-v2', 
            'deberta-base': 'microsoft/deberta-base', 
            'xlm-roberta-base': 'xlm-roberta-base'
            }

        if model_name in models_path:
            print(f"\nRetrieving model path for {model_name}\nModel path: {models_path[model_name]}")
            return models_path[model_name]
        else:
            raise ValueError(f"{model_name} not found in models_path.")    

    def set_model(self, model_name):
        print("\nSetting up TF model: tokenizer and model.")
        print(f"Model: {model_name}")
        model_path = self.tf_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.transformer_model = TFAutoModel.from_pretrained(model_path)

    def transform_X_data(self, X):

        # Tokenize the data using the transformer's tokenizer
        input_ids, attention_mask = self._tokenizer_vectorizer(X, self.max_length)

        # Concatenate input_ids and attention_mask along the second axis to form X
        X = np.concatenate((input_ids, attention_mask), axis=1)

        return X

    def _tokenizer_vectorizer(self, texts, max_length):
        input_ids = []
        attention_mask = []
        for text in texts:
            inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, 
                                                padding='max_length', return_attention_mask=True, 
                                                return_token_type_ids=False, truncation=True)
            input_ids.append(inputs['input_ids'])
            attention_mask.append(inputs['attention_mask'])

        return np.array(input_ids), np.array(attention_mask)

    def train(self, X_train, y_train, X_val, y_val, model_tuple, num_classes, loss):
        print("\nLoss in train:", loss)
        model_name, model, hyperparams = model_tuple

        # Transform X_train data
        X_train = self.transform_X_data(X_train)
        X_val = self.transform_X_data(X_val)

        # Process labels for training
        print(f"\ny_train (before encoding):\n{y_train}")
        if num_classes == math.inf: # Regression
            print(f"\ny_train no encoding, for regression")
            y_train_processed = y_train  # For regression, no encoding needed
            y_val_processed = y_val
        elif num_classes > 2:  # Multiclass Classification
            # Apply one-hot encoding for neural networks in case of multiclass classification task
            # Ensure the OneHotEncoder was fit on the whole set of labels 'y', not just y_train
            encoded_y_train = self.label_encoder.transform(y_train) # Encoding labels
            y_train_processed = to_categorical(encoded_y_train) # One-hot encoding

            encoded_y_val = self.label_encoder.transform(y_val)
            y_val_processed = to_categorical(encoded_y_val)

            print(f"\ny_train is label encoded and one-hot encoded , for multiclass")
        else:  # Binary Classification
            y_train_processed = self.label_encoder.transform(y_train) # Encoding labels
            y_val_processed = self.label_encoder.transform(y_val)

            print(f"\ny_train is label encoded, for binary")

        # Ensure X_train and y_train have the same length
        assert len(X_train) == len(y_train)
        
        # Split X_train into input_ids and attention_mask
        input_ids, attention_mask = np.split(X_train, 2, axis=1)
        val_input_ids, val_attention_mask = np.split(X_val, 2, axis=1)

        #default learning rate 0.001 (optimizer='adam')
        # if not converging: change from 0.001 to 0.002, 0.005
        # converging too quickly: change to 0.0005, 0.0001
        #optimizer = Adam(learning_rate=0.01)
        optimizer = Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, weight_decay=1e-2)

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        start_time = time.time()
        #history=  model.fit([input_ids, attention_mask], y_train_processed, epochs=self.epochs, batch_size=self.batch_size)
        #"""
        history = model.fit(
            [input_ids, attention_mask], 
            y_train_processed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=([val_input_ids, val_attention_mask], y_val_processed)
        )
        #"""
        train_time = time.time() - start_time
        print(f"Model trained in {train_time:.3f} seconds\n")
        
        return model, train_time, history
    
    # Converts the raw predictions to class labels
    def raw_predictions_to_labels(self, raw_predictions, num_classes):
        if num_classes > 2:
            # Multiclass classification with one output neuron per class, use argmax
            predictions = np.argmax(raw_predictions, axis=-1)
        elif num_classes == 2:
            # Binary classification with one output neuron, threshold at 0.5
            predictions = (raw_predictions > 0.5).astype(int).flatten()
    
        return self.label_encoder.inverse_transform(predictions)

    def predict(self, model, X_test, num_classes):

        # Transform X_test data
        X_test = self.transform_X_data(X_test)
        
        # Split X_test into input_ids and attention_mask
        input_ids, attention_mask = np.split(X_test, 2, axis=1)

        start_time = time.time()
        raw_predictions = model.predict([input_ids, attention_mask])
        predict_time = time.time() - start_time
        print(f"Prediction in {predict_time:.3f} seconds\n")

        if num_classes != math.inf: # Classification task (binary, multiclass), convert raw predictions to labels/values
            predictions = self.raw_predictions_to_labels(raw_predictions, num_classes)
        else: # Regression task, raw_predictions are already the predicted values
            predictions = raw_predictions

        return predictions, predict_time

    @staticmethod
    def get_new_model_dir():
        base_dir = 'saved_models/exp'
        existing_dirs = glob.glob(f"{base_dir}*")  # list existing directories

        if not existing_dirs:
            # If no directories exist yet, start with "exp1"
            return base_dir + '1'
        else:
            # If there are existing directories, find the highest index and increment
            highest_index = max([int(dir.split('exp')[-1]) for dir in existing_dirs])
            return base_dir + str(highest_index + 1)


    def save_model(self, model, model_name, dataset_name, exp_dir):
        full_dir_path = f"{exp_dir}/Transformers"
        os.makedirs(full_dir_path, exist_ok=True)
        filename = f'{full_dir_path}/{model_name}-{dataset_name}.h5'
        save_model(model, filename)


    """
    def load_model(self, model_name, exp_dir):
        filename = f'{exp_dir}/Transformers/{model_name}.h5'
        model = load_model(filename, custom_objects={'TFAutoModel': TFAutoModel})
        return model
    """

    def update_model_with_best_params(self, model_func, best_params, loss):
        
        model = model_func
        # Extract the best parameters
        best_learning_rate = best_params.get('learning_rate')

        # Re-compile the model with the best parameters
        model.compile(optimizer=Adam(learning_rate=best_learning_rate),
                    loss=loss,
                    metrics=['accuracy'])

        return model
    

    def build_model(self, model_tuple, loss, hp):
        print("\nLoss in build_model:", loss)
        model_name, model, params = model_tuple

        # Define hyperparameters
        learning_rate = hp.Choice('learning_rate', values=params['learning_rate'])

        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss=loss, 
                  metrics=['accuracy'])

        return model
 
    def hyperparameter_tuning(self, model_tuple, X_train, y_train, is_continuous, loss, num_classes, method='random_search'):
        
        # Transform X_train data
        X_train = self.transform_X_data(X_train)
        
        print("\nLoss in hyperparameter_tuning:", loss)
        print(f"X_train:\n{X_train}")
        print(f"y_train:\n{y_train}")

        input_ids, attention_mask = np.split(X_train, 2, axis=1)
        _, _, params = model_tuple


        # Process labels for hyperparameter tuning
        print(f"\nProcessing labels for hyperparameter tuning..")
        if num_classes == math.inf: # Regression
            print(f"\ny_train no encoding, for regression")
            y_train_processed = y_train  # For regression, no encoding needed
        elif num_classes > 2:  # Multiclass Classification
            encoded_y = self.label_encoder.transform(y_train) # Encoding labels
            y_train_processed = to_categorical(encoded_y) # One-hot encoding
            print(f"\ny_train is label encoded and one-hot encoded , for multiclass")
        else:  # Binary Classification
            y_train_processed = self.label_encoder.transform(y_train) # Encoding labels
            print(f"\ny_train is label encoded, for binary")
        print(f"\ny_train_processed =\n{y_train_processed}")

        tuner = RandomSearch(
            lambda hp: self.build_model(model_tuple, loss, hp),  # using lambda function to pass model_tuple to build_model
            objective='val_accuracy',
            max_trials=5,  # how many model variations to test?
            executions_per_trial=3, # how many trials per variation? (same model could perform differently)
            directory='random_search',
            project_name='sentiment_analysis'
        )

        print("\nBuilt tuner.")

        print("\nTuner search space summary...")
        tuner.search_space_summary()

        # Start the search for the best hyperparameters
        for batch_size in params['batch_size']:
            tuner.search([input_ids, attention_mask], y_train_processed, 
                        epochs=5, 
                        batch_size=batch_size,
                        validation_split=0.2)
            
        print(f"Finished hyperparameter search")

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        return best_hps