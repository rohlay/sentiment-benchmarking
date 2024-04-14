import math
import json
# Machine Learning Libraries
from sklearn import naive_bayes, svm, tree, ensemble, linear_model, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, GRU, SimpleRNN

# Transformer Libraries
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

from sentiment_analysis_project.scripts.config_dir import CONFIG_ML


class AllModels:
    def __init__(self, num_features, num_classes, is_continuous):
        with open(CONFIG_ML, 'r') as f:
            self.config = json.load(f)

        # Fetch level for DL models from the config, default to 1 if not present
        self.level = self.config['modules']['Deep Learning'].get('level', 1)

        self.num_features = num_features
        self.num_classes = num_classes
        self.is_continuous = is_continuous
        self.models = {}
        if self.config['modules']['Machine Learning']['active']:
            self.models['Machine Learning'] = self._get_models_ml()
        if self.config['modules']['Deep Learning']['active']:
            self.models['Deep Learning'] = self._get_models_dl()
        if self.config['modules']['Transformers']['active']:
            self.models['Transformers'] = self._get_models_transformer()

    def get_models(self, module_key):
        with open(CONFIG_ML, 'r') as f:
            config = json.load(f)
        # Get all models for the given module_key
        all_models_for_module = self.models.get(module_key)
        # Filter these models based on the configuration in the JSON file
        # model[0] is the model name, that is found in the json file
        active_models_for_module = [model for model in all_models_for_module if config['modules'][module_key]['models'][model[0]]]
        #active_models_for_module = [(model_name, model_instance, hyperparams) for model_name, model_instance, hyperparams in all_models_for_module if config['modules'][module_key]['models'][model_name]]


        return active_models_for_module

    ###################################################################################################

    ### TRANSFORMER MODELS

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

    def create_transformer_model(self, model_name, num_classes, dropout_rate=0.35):
        print(f"\nInitializing Transformers in All Models")
        print("Creating TF models and configuring architecture...")
        model_path = self.tf_model_path(model_name)
        transformer_model = TFAutoModel.from_pretrained(model_path)
        input_word_ids = Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(128,), dtype=tf.int32, name="input_mask")
        
        # Get the sequence output from the transformer
        transformer_output = transformer_model([input_word_ids, input_mask])[0]
        
        # Select the output of the first token ([CLS] token)
        cls_output = transformer_output[:, 0, :]
        
        # Apply dropout to the output of the [CLS] token
        dropout_layer = Dropout(dropout_rate)(cls_output)

        # Determine the number of output neurons and the activation function
        if num_classes == math.inf:  # If num_classes is infinity, it's a regression problem
            output_neurons = 1
            activation = 'linear'  # Use linear activation for regression problems
        elif num_classes > 2:
            output_neurons = num_classes
            activation = 'softmax'  # Use softmax for multiclass classification problems
        else:  # Binary classification
            output_neurons = 1
            activation = 'sigmoid'  # Use sigmoid for binary classification problems

        # Create the output layer with the specified number of neurons and activation function
        output = Dense(output_neurons, activation=activation)(dropout_layer)

        print(f"\nFinal layer activation: {activation}")
        # Create the model
        model = Model(inputs=[input_word_ids, input_mask], outputs=output)
        return model

    """
    def create_transformer_model(self, model_name, num_classes):
        print(f"\nInitializing Transformers in All Models")
        print("Creating TF models and configuring arquitecture...")
        model_path = self.tf_model_path(model_name)
        transformer_model = TFAutoModel.from_pretrained(model_path)
        input_word_ids = Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(128,), dtype=tf.int32, name="input_mask")
        transformer_layer = transformer_model([input_word_ids, input_mask])[0]

        if num_classes == math.inf:  # If num_classes is infinity, it's a regression problem
            output_neurons = 1
            activation = 'linear'  # Use linear activation for regression problems
            output = Dense(output_neurons, activation=activation)(transformer_layer[:, 0, :])
        elif num_classes > 2:
            activation = 'softmax'  # Use softmax for multiclass classification problems
            output_neurons = num_classes
            output = Dense(output_neurons, activation=activation)(transformer_layer[:, 0, :])
        else:
            output_neurons = 1
            activation = 'sigmoid'  # Use sigmoid for binary classification problems
            output = Dense(output_neurons, activation=activation)(transformer_layer[:, 0, :])


        print(f"\nFinal layer activation: {activation}")
        model = Model(inputs=[input_word_ids, input_mask], outputs=output)
        return model
    """

    def _get_models_transformer(self):
        models = [  'bert-base-uncased',
                    'bert-base-multilingual-cased',
                    'distilbert-base-uncased',
                    'roberta-base', 
                    'robertuito-sentiment-analysis',
                    'electra-base-discriminator',
                    'albert-base-v2', 
                    'deberta-base', 
                    'xlm-roberta-base']
        
        hyperparams = self._get_hyperparams_transformer()

        # Load the JSON configuration
        with open(CONFIG_ML, 'r') as f:
            config = json.load(f)
        active_models = config['modules']['Transformers']['models']

        models_transformer = []
        for model_name in models:
            # Check if the model is active based on the JSON configuration
            if active_models.get(model_name):
                try:
                    print(f"\nCreating model tuple for: {model_name}")
                    model = self.create_transformer_model(model_name, self.num_classes)
                    model_tuple = (model_name, model, hyperparams[model_name])
                    models_transformer.append(model_tuple)
                    print(model_tuple)
                    print()
                except Exception as e:
                    print(f"Error creating model tuple for {model_name}: {e}")
        
        return models_transformer
   
    def _get_hyperparams_transformer(self):
        # Define the hyperparameters for your transformer models
        hyperparams = {
            'bert-base-uncased': {'learning_rate': [1e-2, 1e-3, 1e-4], 'batch_size': [16, 32, 64]},
            'distilbert-base-uncased': {'learning_rate': [1e-2, 1e-3, 1e-4], 'batch_size': [16, 32, 64]},
            'roberta-base': {'learning_rate': [1e-2, 1e-3, 1e-4], 'batch_size': [16, 32, 64]},
            'robertuito-sentiment-analysis': {'learning_rate': [1e-2, 1e-3, 1e-4], 'batch_size': [16, 32, 64]},
            'electra-base-discriminator': {'learning_rate': [1e-2, 1e-3, 1e-4], 'batch_size': [16, 32, 64]},
            'albert-base-v2': {'learning_rate': [1e-2, 1e-3, 1e-4], 'batch_size': [16, 32, 64]},
            'deberta-base': {'learning_rate': [1e-2, 1e-3, 1e-4], 'batch_size': [16, 32, 64]},
            'xlm-roberta-base': {'learning_rate': [1e-2, 1e-3, 1e-4], 'batch_size': [16, 32, 64]}
        }
        
        return hyperparams

    ###################################################################################################

    ### MACHINE LEARNING MODELS

    def _get_hyperparams_ml(self):
        # Define the hyperparameters for your machine learning models
        hyperparams = {
            'Naive Bayes': {},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},    # {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001]}
            'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_leaf': [1, 2, 4]},
            'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30], 'min_samples_leaf': [1, 2, 4]},
            'Logistic Regression': {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            'Linear Regression': {},  # No hyperparameters to tune
            'Gradient Boosting': {'n_estimators': [10, 50, 100], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [3, 5, 7, 9]},
            'KNN': {'n_neighbors': [3, 5, 7, 9, 11]},
            'ANN': {'alpha': [0.0001, 0.05], 'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)], 'learning_rate': ['constant','adaptive']}
        }
        return hyperparams

    def _get_models_ml(self):
        hyperparams = self._get_hyperparams_ml()
        if self.is_continuous:
            models = [
                #('Naive Bayes', GaussianNB(), hyperparams['Naive Bayes']), Doesn't support regression
                ('SVM', svm.SVR(), hyperparams['SVM']),
                ('Decision Tree', tree.DecisionTreeRegressor(), hyperparams['Decision Tree']),
                ('Random Forest', RandomForestRegressor(), hyperparams['Random Forest']),
                ('Linear Regression', linear_model.LinearRegression(), hyperparams['Linear Regression']),
                ('Gradient Boosting', GradientBoostingRegressor(), hyperparams['Gradient Boosting']),
                ('KNN', KNeighborsRegressor(), hyperparams['KNN']),
                ('ANN', MLPRegressor(random_state=1, max_iter=500), hyperparams['ANN'])
            ]
        else:
            models = [
                ('Naive Bayes', naive_bayes.MultinomialNB(), hyperparams['Naive Bayes']),
                ('SVM', svm.SVC(), hyperparams['SVM']),
                ('Decision Tree', tree.DecisionTreeClassifier(), hyperparams['Decision Tree']),
                ('Random Forest', RandomForestClassifier(), hyperparams['Random Forest']),
                ('Logistic Regression', linear_model.LogisticRegression(), hyperparams['Linear Regression']),
                ('Gradient Boosting', GradientBoostingClassifier(), hyperparams['Gradient Boosting']),
                ('KNN', KNeighborsClassifier(), hyperparams['KNN']),
                ('ANN', MLPClassifier(random_state=1, max_iter=500), hyperparams['ANN'])
            ]

        return models
    
    ###################################################################################################

    # DEEP LEARNING MODELS

    def _get_hyperparams_dl(self):
        # Define the hyperparameters for your deep learning models
        hyperparams = {
            'MLP': {'learning_rate': [0.01, 0.001, 0.0001], 'neurons': [16, 32, 64, 128]}, 
            'CNN': {'learning_rate': [0.01, 0.001, 0.0001], 'neurons': [16, 32, 64, 128]},  
            'RNN': {'learning_rate': [0.01, 0.001, 0.0001], 'neurons': [16, 32, 64, 128]},  
            'LSTM': {'learning_rate': [0.01, 0.001, 0.0001], 'neurons': [16, 32, 64, 128]},  
            'GRU': {'learning_rate': [0.01, 0.001, 0.0001], 'neurons': [16, 32, 64, 128]}
        }
        return hyperparams

    def _append_hyperparams(self, models, hyperparams):
        # Append the hyperparameters to the model tuples
        return [(model_name, model_func, hyperparams.get(model_name, {})) for model_name, model_func in models]

    def get_dynamic_models(self, output_neurons, output_activation):
        if self.level == 1:
            models = [
                ("MLP", Sequential([
                    Dense(32, activation='relu'),
                    Dropout(0.25),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("CNN", Sequential([
                    Conv1D(32, 3, activation='relu', input_shape=(self.num_features, 1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(32, 3, activation='relu'),
                    GlobalMaxPooling1D(),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("RNN", Sequential([
                    SimpleRNN(16, return_sequences=True, input_shape=(self.num_features, 1)),
                    SimpleRNN(16),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("LSTM", Sequential([
                    LSTM(16, return_sequences=True, input_shape=(self.num_features, 1)),
                    LSTM(16),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("GRU", Sequential([
                    GRU(16, return_sequences=True, input_shape=(self.num_features, 1)),
                    GRU(16),
                    Dense(output_neurons, activation=output_activation)
                ]))
            ]
        elif self.level == 2:
            models = [
                ("MLP", Sequential([
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("CNN", Sequential([
                    Conv1D(64, 3, activation='relu', input_shape=(self.num_features, 1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(64, 3, activation='relu'),
                    GlobalMaxPooling1D(),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("RNN", Sequential([
                    SimpleRNN(32, return_sequences=True, input_shape=(self.num_features, 1)),
                    SimpleRNN(32, return_sequences=True),
                    SimpleRNN(32),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("LSTM", Sequential([
                    LSTM(32, return_sequences=True, input_shape=(self.num_features, 1)),
                    LSTM(32, return_sequences=True),
                    LSTM(32),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("GRU", Sequential([
                    GRU(32, return_sequences=True, input_shape=(self.num_features, 1)),
                    GRU(32, return_sequences=True),
                    GRU(32),
                    Dense(output_neurons, activation=output_activation)
                ]))
            ]
        elif self.level == 3:
            models = [
                ("MLP", Sequential([
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("CNN", Sequential([
                    Conv1D(128, 3, activation='relu', input_shape=(self.num_features, 1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(128, 3, activation='relu'),
                    GlobalMaxPooling1D(),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("RNN", Sequential([
                    SimpleRNN(64, return_sequences=True, input_shape=(self.num_features, 1)),
                    SimpleRNN(64, return_sequences=True),
                    SimpleRNN(64, return_sequences=True),
                    SimpleRNN(64),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("LSTM", Sequential([
                    LSTM(64, return_sequences=True, input_shape=(self.num_features, 1)),
                    LSTM(64, return_sequences=True),
                    LSTM(64, return_sequences=True),
                    LSTM(64),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("GRU", Sequential([
                    GRU(64, return_sequences=True, input_shape=(self.num_features, 1)),
                    GRU(64, return_sequences=True),
                    GRU(64, return_sequences=True),
                    GRU(64),
                    Dense(output_neurons, activation=output_activation)
                ]))
            ]
        elif self.level == 4:
            models = [
                ("MLP", Sequential([
                    Dense(256, activation='relu'),
                    Dropout(0.5),
                    Dense(256, activation='relu'),
                    Dropout(0.5),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("CNN", Sequential([
                    Conv1D(256, 3, activation='relu', input_shape=(self.num_features, 1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(256, 3, activation='relu'),
                    GlobalMaxPooling1D(),
                    Dense(256, activation='relu'),
                    Dropout(0.5),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("RNN", Sequential([
                    SimpleRNN(128, return_sequences=True, input_shape=(self.num_features, 1)),
                    SimpleRNN(128, return_sequences=True),
                    SimpleRNN(128, return_sequences=True),
                    SimpleRNN(128, return_sequences=True),
                    SimpleRNN(128),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("LSTM", Sequential([
                    LSTM(128, return_sequences=True, input_shape=(self.num_features, 1)),
                    LSTM(128, return_sequences=True),
                    LSTM(128, return_sequences=True),
                    LSTM(128, return_sequences=True),
                    LSTM(128),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("GRU", Sequential([
                    GRU(128, return_sequences=True, input_shape=(self.num_features, 1)),
                    GRU(128, return_sequences=True),
                    GRU(128, return_sequences=True),
                    GRU(128, return_sequences=True),
                    GRU(128),
                    Dense(output_neurons, activation=output_activation)
                ]))
            ]
        elif self.level == 5:
            models = [
                ("MLP", Sequential([
                    Dense(512, activation='relu'),
                    Dropout(0.5),
                    Dense(512, activation='relu'),
                    Dropout(0.5),
                    Dense(256, activation='relu'),
                    Dropout(0.5),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("CNN", Sequential([
                    Conv1D(512, 3, activation='relu', input_shape=(self.num_features, 1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(512, 3, activation='relu'),
                    GlobalMaxPooling1D(),
                    Dense(512, activation='relu'),
                    Dropout(0.5),
                    Dense(256, activation='relu'),
                    Dropout(0.5),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("RNN", Sequential([
                    SimpleRNN(256, return_sequences=True, input_shape=(self.num_features, 1)),
                    SimpleRNN(256, return_sequences=True),
                    SimpleRNN(256, return_sequences=True),
                    SimpleRNN(256, return_sequences=True),
                    SimpleRNN(256, return_sequences=True),
                    SimpleRNN(256),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("LSTM", Sequential([
                    LSTM(256, return_sequences=True, input_shape=(self.num_features, 1)),
                    LSTM(256, return_sequences=True),
                    LSTM(256, return_sequences=True),
                    LSTM(256, return_sequences=True),
                    LSTM(256, return_sequences=True),
                    LSTM(256),
                    Dense(output_neurons, activation=output_activation)
                ])),
                ("GRU", Sequential([
                    GRU(256, return_sequences=True, input_shape=(self.num_features, 1)),
                    GRU(256, return_sequences=True),
                    GRU(256, return_sequences=True),
                    GRU(256, return_sequences=True),
                    GRU(256, return_sequences=True),
                    GRU(256),
                    Dense(output_neurons, activation=output_activation)
                ]))
            ]
        else:
            raise ValueError("Invalid level provided. Please select a level between 1 and 5.")
        
        # Append the hyperparameters
        hyperparams = self._get_hyperparams_dl()
        models = self._append_hyperparams(models, hyperparams)

        return models

    def _get_models_dl(self):
        
        # Update output_activation based on the number of classes
        if self.is_continuous:
            output_neurons = 1
            output_activation = 'linear'
            print(f"\nContinuous values, output activation is {output_activation}")
        elif self.num_classes > 2:
            output_neurons = self.num_classes
            output_activation = 'softmax' 
            print(f"\nMulticlass, output activation is {output_activation}")
        else:
            output_neurons = 1
            output_activation = 'sigmoid'
            print(f"\nBinary class, output activation is {output_activation}")

        print(f"Output neurons: {output_neurons}")
        # Load model configurations
        models_raw = self.get_dynamic_models(output_neurons, output_activation)
        models = [(model_name, model, hyperparams) for model_name, model, hyperparams in models_raw]  # format dynamic models

        return models
    