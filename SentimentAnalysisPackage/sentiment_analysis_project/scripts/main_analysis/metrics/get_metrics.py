from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def binary_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def multiclass_auc(y_true, y_pred):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')


def get_binary_metrics():
    return {
        'accuracy_score': accuracy_score,
        'precision_score': lambda y_true, y_pred: precision_score(y_true, y_pred, pos_label=max(set(y_true)), zero_division=0),
        'recall_score': lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=max(set(y_true)), zero_division=0),
        'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=max(set(y_true)), zero_division=0),
        'roc_auc_score': lambda y_true, y_pred: binary_auc(y_true, y_pred)
    }


def get_multiclass_metrics():
    return {
        'accuracy_score': accuracy_score,
        'precision_score': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_score': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'roc_auc_score': lambda y_true, y_pred: multiclass_auc(y_true, y_pred)
    }


def get_continuous_metrics():
    return {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'root_mean_squared_error': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2_score': r2_score,
        'explained_variance_score': explained_variance_score
    }

"""
def get_normalized_continuous_metrics():
    return {
        'mean_squared_error': lambda y_true, y_pred, y_max, y_min: mean_squared_error(y_true, y_pred) / (y_max - y_min) ** 2,
        'mean_absolute_error': lambda y_true, y_pred, y_max, y_min: mean_absolute_error(y_true, y_pred) / (y_max - y_min),
        'root_mean_squared_error': lambda y_true, y_pred, y_max, y_min: np.sqrt(mean_squared_error(y_true, y_pred)) / (y_max - y_min),
        'r2_score': lambda y_true, y_pred: r2_score(y_true, y_pred),
        'explained_variance_score': lambda y_true, y_pred: explained_variance_score(y_true, y_pred)
    }
"""