import os
import time
import json
import math
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import tensorflow as tf
from keras.models import load_model
from kerastuner.tuners import RandomSearch, Hyperband
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, GRU, SimpleRNN
from tensorflow.keras.models import Sequential

from sentiment_analysis_project.scripts.config_dir import GLOVE_ENGLISH_FILE_PATH, GLOVE_SPANISH_FILE_PATH, CONFIG_ML


class DeepLearningModule:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.glove_model = None
        self.num_features = None

        # Initialize the parameters needed to build the model as None
        self.model_class = None
        self.layers = None
        self.param_grid = None
        self.loss = None
        self.num_classes = None

        with open(CONFIG_ML, 'r') as f:
            config = json.load(f)
        deeplearning_config = config['modules']['Deep Learning']
        self.epochs = deeplearning_config['epochs']
        self.batch_size = deeplearning_config['batch_size']

    def load_glove_model(self, language):
        print(f"\nLoading {language} GloVe model...(in DL module)")
        glove_file, self.num_features, no_header = self.set_language_dependent_attributes(language)
        self.glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=no_header)
        print("GloVe model loaded...\n")
        return self.num_features

    def set_language_dependent_attributes(self, language):
        if language == 'english':
            glove_file = GLOVE_ENGLISH_FILE_PATH
            num_features = 100
            no_header = True
        elif language == 'spanish':
            glove_file = GLOVE_SPANISH_FILE_PATH
            num_features = 300
            no_header = False
        else:
            raise ValueError(f"Unknown language: {language}")
        return glove_file, num_features, no_header


    def load_data(self, dataset):
        df = pd.read_csv(dataset)
        X = df['text']
        y = df['sentiment']
        self.label_encoder.fit(y) 

        return X, y

    def average_word_vectors(self, words, model):
        """
        Averages word vectors for a set of words.
        """
        feature_vector = np.zeros((self.num_features,), dtype="float64")
        nwords = 0.
        for word in words:
            try: #  if word is in the model's vocabulary 
                feature_vector = np.add(feature_vector, model[word])
                nwords += 1
            except KeyError:
                pass
        if nwords > 0:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector

    def averaged_word_vectorizer(self, corpus, model):
        """
        Vectorize a set of words using their average word vectors.
        """
        features = [self.average_word_vectors(tokenized_sentence, model) for tokenized_sentence in corpus]
        return np.array(features)

    def glove_vectorizer(self, X_train):
        """
        Vectorize a set of words using GloVe.
        """
        train_word_list = X_train.apply(word_tokenize)
        X_train = self.averaged_word_vectorizer(corpus=train_word_list, model=self.glove_model)
        return X_train

    def train(self, X_train, y_train, X_val, y_val, model_tuple, num_classes, loss):
        
        model_name, model, _ = model_tuple
        # Process labels for training
        print(f"\ny_train (before encoding):\n{y_train}")
        print(f"\nProcessing labels for training....")
        if num_classes == math.inf: # Regression
            print(f"y_train no encoding, for regression")
            y_train_processed = y_train  # For regression, no encoding needed
            y_val_processed = y_val
        elif num_classes > 2:  # Multiclass Classification
            # Apply one-hot encoding for neural networks in case of multiclass classification task
            # Ensure the OneHotEncoder was fit on the whole set of labels 'y', not just y_train
            encoded_y_train = self.label_encoder.transform(y_train) # Encoding labels
            y_train_processed = to_categorical(encoded_y_train) # One-hot encoding

            encoded_y_val = self.label_encoder.transform(y_val)
            y_val_processed = to_categorical(encoded_y_val)
            print(f"y_train is label encoded and one-hot encoded , for multiclass")
        else:  # Binary Classification
            y_train_processed = self.label_encoder.transform(y_train) # Encoding labels
            y_val_processed = self.label_encoder.transform(y_val)
            print(f"y_train is label encoded, for binary")

        print(f"\ny_train_processed =\n{y_train_processed}")


        # Ensure X_train and y_train have the same length
        assert len(X_train) == len(y_train)
        # Perform GloVe vectorization
        X_train_processed = self.glove_vectorizer(X_train)
        X_val_processed = self.glove_vectorizer(X_val)

        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        start_time = time.time()
        # add epochs, batch size from json
        #model.fit(X_train_processed, y_train_processed, epochs=20, batch_size=32)
        #history = model.fit(X_train_processed, y_train_processed,epochs=self.epochs, batch_size=self.batch_size)
        #"""
        history = model.fit(
            X_train_processed,
            y_train_processed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val_processed, y_val_processed)
        )
        #"""
        train_time = time.time() - start_time
        print(f"Model trained in {train_time:.3f} seconds\n")

        # After training, save the model
        #trained_model = model
        #self.save_model(trained_model, model_name)

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

        X_test_processed = self.glove_vectorizer(X_test)

        start_time = time.time()
        raw_predictions = model.predict(X_test_processed)
        predict_time = time.time() - start_time
        print(f"Prediction in {predict_time:.3f} seconds\n")
        
        if num_classes != math.inf: # Classification task (binary, multiclass), convert raw predictions to labels/values
            predictions = self.raw_predictions_to_labels(raw_predictions, num_classes)
        else: # Regression task, raw_predictions are already the predicted values
            predictions = raw_predictions.flatten()

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
        full_dir_path = f"{exp_dir}/DL"
        os.makedirs(full_dir_path, exist_ok=True)
        filename = f'{full_dir_path}/{model_name}-{dataset_name}.h5'
        model.save(filename)

    """
    def load_model(self, model_name):
        filename = f'saved_models/{model_name}.h5'
        loaded_model = load_model(filename)
        return loaded_model
    """
        
    def build_model_with_best_params(self, best_params):
        return self.build_model(best_params, self.model_class, self.layers, self.param_grid, self.loss, self.num_classes)

    def build_model(self, hp, model_class, layers, param_grid, loss, num_classes):
        # initialize the model
        model = model_class()
        # Add each layer to the model
        for i, layer in enumerate(layers):
            # Check the type of the layer 
            if isinstance(layer, keras.layers.Dense): 

                # If it's the last layer and a Dense layer, set the appropriate number of units
                if i == len(layers)-1:  # Check if it's the last layer
                    # Here we check the number of classes to set the correct number of output neurons and activation
                    if num_classes == math.inf:  # Regression
                        model.add(keras.layers.Dense(units=1, activation='linear'))  
                    elif num_classes > 2:  # Multiclass Classification
                        model.add(keras.layers.Dense(units=num_classes, activation='softmax'))  
                    else:  # Binary Classification
                        model.add(keras.layers.Dense(units=1, activation='sigmoid'))  

                else: # If it's a Dense layer or Conv1D layer, we will tune the number of neurons/filters
                    hp_units = hp.Int('dense_units_' + str(i), min_value=param_grid['neurons'][0], max_value=param_grid['neurons'][-1], step=32)
                    model.add(keras.layers.Dense(units=hp_units, activation=layer.activation))

            elif isinstance(layer, keras.layers.Conv1D):
                hp_filters = hp.Int('conv1D_filters_' + str(i), min_value=param_grid['neurons'][0], max_value=param_grid['neurons'][-1], step=32)
                model.add(keras.layers.Conv1D(filters=hp_filters, kernel_size=layer.kernel_size, activation=layer.activation, input_shape=layer.input_shape[1:]))
            elif isinstance(layer, (keras.layers.SimpleRNN, keras.layers.LSTM, keras.layers.GRU)):
                hp_units = hp.Int('rnn_units_' + str(i), min_value=param_grid['neurons'][0], max_value=param_grid['neurons'][-1], step=32)
                model.add(type(layer)(units=hp_units, return_sequences=layer.return_sequences, input_shape=(self.num_features, 1)))
            else:
                model.add(layer)

        # Here we define hyperparameters for tuning within the model building function.
        hp_learning_rate = hp.Choice('learning_rate', param_grid['learning_rate'])

        # compile the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss=loss,
                    metrics=['accuracy'])
        return model


    def hyperparameter_tuning(self, model_tuple, X_train, y_train, is_continuous, loss, num_classes, method='random_search'):

        model_name, model_instance, param_grid = model_tuple  # unpacking model tuple 
        print(f"\nStarting hyperparameter tuning on model: {model_name} using {method}")
        layers = model_instance.layers  # extract the layers from the Sequential model

        # Store the necessary parameters as attributes
        self.model_class = type(model_instance)
        self.layers = model_instance.layers
        self.param_grid = param_grid
        self.loss = loss
        self.num_classes = num_classes

        if method == 'random_search':
            print("Performing Randomized Search...")
            tuner = RandomSearch(
                lambda hp: self.build_model(hp, type(model_instance), layers, param_grid, loss, num_classes),
                objective='val_accuracy',
                max_trials=5,  # how many model variations to test?
                executions_per_trial=3,  # how many trials per variation? (same model could perform differently)
                directory='keras_tuner',
                project_name='sentiment_analysis'
            )
        elif method == 'grid_search':
            print("Performing Grid Search...")
            tuner = Hyperband(
                lambda hp: self.build_model(hp, type(model_instance), layers, param_grid, loss, num_classes),
                objective='val_accuracy',
                max_epochs=5,
                directory='keras_tuner',
                project_name='sentiment_analysis'
            )
        else:
            raise ValueError("Invalid method provided. Please select either 'random_search' or 'grid_search'.")

        print("\nTuner search space summary...")
        tuner.search_space_summary()

        # Process labels for hyperparameter tuning
        print(f"\nProcessing labels for hiperparameter tuning....")
        if num_classes == math.inf: # Regression
            print(f"y_train no encoding, for regression")
            y_train_processed = y_train  # For regression, no encoding needed
        elif num_classes > 2:  # Multiclass Classification
            # Apply one-hot encoding for neural networks in case of multiclass classification task
            # Ensure the OneHotEncoder was fit on the whole set of labels 'y', not just y_train
            encoded_y = self.label_encoder.transform(y_train) # Encoding labels
            y_train_processed = to_categorical(encoded_y) # One-hot encoding
            print(f"y_train is label encoded and one-hot encoded , for multiclass")
        else:  # Binary Classification
            y_train_processed = self.label_encoder.transform(y_train) # Encoding labels
            print(f"y_train is label encoded, for binary")

        # Ensure X_train and y_train have the same length
        assert len(X_train) == len(y_train)
        # Perform GloVe vectorization
        X_train_processed = self.glove_vectorizer(X_train)

        # Perform hypertuning
        tuner.search(X_train_processed, y_train_processed,
                    epochs=5,
                    validation_split=0.2)
        
        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"Finished Hyperparameter tuning.")

        return best_hps