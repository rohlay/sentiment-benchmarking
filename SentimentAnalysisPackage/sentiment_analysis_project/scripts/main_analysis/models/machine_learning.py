import os
import math
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from tensorflow.keras.utils import to_categorical
import glob
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sentiment_analysis_project.scripts.config_dir import GLOVE_ENGLISH_FILE_PATH, GLOVE_SPANISH_FILE_PATH


class MachineLearningModule:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        #self.one_hot_encoder = OneHotEncoder()
        self.glove_model = None
        self.num_features = None
        self.unique_labels = None  # Store unique labels

    def load_glove_model(self, language):
        print(f"\nLoading {language} GloVe model...(in ML module)")
        glove_file, self.num_features, no_header = self.set_language_dependent_attributes(language)
        self.glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=no_header)
        print("GloVe model loaded...")
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
 
        # Encoding labels
        # Important: encodes nClasses as integers from 0 to nClasses-1
        # LabelEncoder simply assigns a unique numeric identifier to each unique class
        # Need to use the complete y data of the dataset to encode to identify all classes
        # (this step can't be done in def train, with a reduced y_train 
        # y_train < y, therefore it won't identify all nClasses
        self.label_encoder.fit(y)
        #self.one_hot_encoder.fit(y.values.reshape(-1, 1)) # Reshape y to have two dimensions
        # Store unique labels for later usage in inverse transform
        self.unique_labels = np.unique(y)

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

    def tfidf_vectorizer(self, X_train):
        """
        Vectorize a set of words using TF-IDF.
        """
        vect = TfidfVectorizer(max_features=5000)
        vect.fit(X_train)
        X_train = vect.transform(X_train)
        return X_train, vect

    def bow_vectorizer(self, X_train):
        """
        Vectorize a set of words using Bag of Words.
        """
        vect = CountVectorizer(binary=True)
        vect.fit(X_train)
        X_train = vect.transform(X_train)
        return X_train, vect

    def glove_vectorizer(self, X_train):
        """
        Vectorize a set of words using GloVe.
        """
        
        train_word_list = X_train.apply(word_tokenize)
        X_train = self.averaged_word_vectorizer(corpus=train_word_list, model=self.glove_model)
        return X_train, (self.glove_model, self.num_features, train_word_list)

    def feature_extraction(self, X_train, feature_type):
        """
        Vectorize a set of words using a specified feature type.
        """
        if feature_type == "bow":
            return self.bow_vectorizer(X_train)
        elif feature_type == "tfidf":
            return self.tfidf_vectorizer(X_train)
        elif feature_type == "glove":
            return self.glove_vectorizer(X_train)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def select_feature_extraction(self, model_name):
        """
        Selects the appropriate feature extraction method based on the model name.
        """
        if model_name in ['Naive Bayes', 'Logistic Regression', 'Linear Regression', 'SVM']:
            return self.feature_extraction, 'tfidf'
        elif model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
            return self.feature_extraction, 'bow'
        elif model_name in ['KNN', 'ANN']:
            return self.feature_extraction, 'glove'
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def train(self, X_train, y_train, X_val, y_val, model_tuple, num_classes, loss):
        """
        Trains a model using a specified feature type.
        """
        model_name, model, _ = model_tuple
        feature_extraction_func, feature_type = self.select_feature_extraction(model_name)
        X_train, vectorizer = feature_extraction_func(X_train, feature_type)
        print(f"\nTraining {model_name} model")


        if num_classes == math.inf: # Regression
            y_train_processed = y_train  # For regression, we do not need to encode the labels
        else:  # Classification
            if model_name == "ANN" and num_classes > 2:
                # Apply one-hot encoding for ANN in case of multiclass classification task only
                # Fit the OneHotEncoder on whole set of labels 'y' not just y_train
                #y_train_processed = self.one_hot_encoder.transform(y_train[:, np.newaxis])
                #y_train_processed = self.one_hot_encoder.transform(y_train.values.reshape(-1, 1))
                encoded_Y = self.label_encoder.transform(y_train) # Encoding labels
                y_train_processed = to_categorical(encoded_Y) # One-hot encoding

            else:
                y_train_processed = self.label_encoder.transform(y_train) # For classification (binary or multiclass), we need to encode the labels
                

        start_time = time.time()
        model.fit(X_train, y_train_processed)
        train_time = time.time() - start_time

        # Save the trained model
        trained_model = (model, vectorizer, feature_type, model_name) 

        # After training, save the model
        #self.save_model(trained_model, model_name)

        print(f"Model trained in {train_time:.3f} seconds")
        return trained_model, train_time

    def raw_predictions_to_labels(self, raw_predictions, model_name):
        # Check if the model is ANN
        if model_name == "ANN":
            if raw_predictions.ndim > 1 and raw_predictions.shape[1] > 1:  # Multiclass classification
                # Multiclass or binary classification with two output neurons, use argmax
                predictions = np.argmax(raw_predictions, axis=-1)
            else:  # Binary classification or single neuron in the output layer
                # Convert the probabilities to class labels based on a threshold of 0.5
                predictions = (raw_predictions > 0.5).astype(int)
        else: 
            # For other models in case of classification, use LabelEncoder's inverse_transform
            predictions = raw_predictions.astype(int)
        
        # Use LabelEncoder's inverse_transform to convert encoded labels back to original form
        return self.label_encoder.inverse_transform(predictions)

    def predict(self, trained_model, X_test, num_classes):
        """
        Predict labels for a set of data using a trained model.
        """
        model, vectorizer, feature_type, model_name = trained_model  

        if feature_type == "glove":
            test_word_list = X_test.apply(word_tokenize)
            X_test_transformed = self.averaged_word_vectorizer(corpus=test_word_list, model=vectorizer[0])
        else:
            X_test_transformed = vectorizer.transform(X_test)

        start_time = time.time()
        raw_predictions = model.predict(X_test_transformed)
        predict_time = time.time() - start_time
        print(f"Prediction in {predict_time:.3f} seconds\n")

        if num_classes == math.inf: # Regression task
            predictions = raw_predictions 
        else: # Classification task: transform labels back to original form
            predictions = self.raw_predictions_to_labels(raw_predictions, model_name)
        
        return predictions, predict_time
    
    def save_model(self, model, model_name, dataset_name, exp_dir):
        full_dir_path = f"{exp_dir}/ML"
        os.makedirs(full_dir_path, exist_ok=True)
        filename = f'{full_dir_path}/{model_name}-{dataset_name}.sav'
        pickle.dump(model, open(filename, 'wb'))

    """
    def load_model(self, model_name):
        filename = f'saved_models/{model_name}.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model
    """ 

    def hyperparameter_tuning(self, model_tuple, X_train, y_train, is_continuous, loss, num_classes, method='random_search'):
        
        model_name, model_func, params = model_tuple  # unpacking model tuple 

        print(f"\nStarting hyperparameter tuning on model: {model_name} using {method}")

        if method == 'grid_search':
            print("Performing Grid Search...")
            search = GridSearchCV(estimator=model_func,
                                param_grid=params,
                                scoring='accuracy',
                                cv=3,
                                n_jobs=-1)
        elif method == 'random_search':
            print("Performing Randomized Search...")
            search = RandomizedSearchCV(estimator=model_func,
                                        param_distributions=params,
                                        scoring='accuracy',
                                        cv=3,
                                        n_jobs=-1)

        # Select the appropriate feature extraction method
        feat_extraction_func, feature_type = self.select_feature_extraction(model_name)
        # Transform X_train using the selected method
        X_train_encoded, _ = feat_extraction_func(X_train, feature_type)

        # Fit search to the training data
        if is_continuous:
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            search.fit(X_train_encoded, y_train_encoded)
        else:
            search.fit(X_train_encoded, y_train)

        # Get the parameters of the best model
        best_params = search.best_params_

        print("Hyperparameter tuning complete.")
        return best_params