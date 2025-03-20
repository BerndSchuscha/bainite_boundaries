import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import random
import inspect
import json
from matplotlib import pyplot as plt
from pathlib import Path
import bainite_boundaries

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def evaluation_conditions_boundaries(y,which_data):
    if which_data=='Ferrite_critCR':
        L=np.logical_and(y<np.log(50),y>np.log(0.1))
    if which_data=='Bainite_start':
        L=np.logical_and(y>200,y<650)
    if which_data=='Martensite_start':
        L=np.logical_and(y>200,y<650)
    if which_data=='Martensite_start_RA':
        L=np.logical_and(y<25)
    if which_data=='Bainite':
        L=np.logical_and(y<25)
    if which_data=='Bainite':
        L=y==1
    return L

def evaluation_conditions(y,y_eva,which_data):
    if which_data=='Ferrite_critCR':
        L=y<np.exp(y_eva[:,1])
    if which_data=='Bainite_start':
        L=y>y_eva[:,0]
    if which_data=='Martensite_start':
        L=y<y_eva[:,0]
    if which_data=='Martensite_start_RA':
        L=y<25
    if which_data=='Bainite':
        L=y<10800
    if which_data=='Austensite':
        L=y==1
    return L



def extend_with_constraints_final(x_test_final, features, features_final):
    
    features = [f.lower() for f in features]
    features_final = [f.lower() for f in features_final]
    features_final_ordered = [f for f in features if f in features_final]
    
    # flag_features = np.array( [f in features_final for f in features] )
    x_test_final_padded = np.zeros((x_test_final.shape[0], len(features)))
    
    for f in features_final_ordered:
        idx_features = features.index(f)
        idx_features_final = features_final.index(f)
        print(f, idx_features, idx_features_final)
        x_test_final_padded[:, idx_features] = x_test_final[:, idx_features_final]
    # x_test_final_padded[:, flag_features] = x_test_final 

    return x_test_final_padded


def extend_with_constraints(x, x_test_final, features, features_final, which_data, alpha=0.5, model_Martensite='RF', model_Bainite='linear'):
    # old version .. should be delete in the feature this takes the austenite grid for the final evaulaiton
    features = [f.lower() for f in features]
    features_final = [f.lower() for f in features_final]
    features_final_ordered = [f for f in features if f in features_final]
    
    # flag_features = np.array( [f in features_final for f in features] )
    x_test_final_padded = np.zeros((x_test_final.shape[0], len(features)))
    
    for f in features_final_ordered:
        idx_features = features.index(f)
        idx_features_final = features_final.index(f)
        print(f, idx_features, idx_features_final)
        x_test_final_padded[:, idx_features] = x_test_final[:, idx_features_final]
    # x_test_final_padded[:, flag_features] = x_test_final 

    if which_data in ['Bainite', 'Ferrite']:
        
        # add two columns for the predictions of Bainite and Martensite start temperatures
        x_test_final_padded = np.concatenate((x_test_final_padded, np.zeros((x_test_final_padded.shape[0], 1))), axis=1)
    
        # load from results the saved predictions
        results_path = Path(str(bainite_boundaries.PROJECT_ROOT), 'bainite_boundaries', 'bainite_boundaries', 'results')
        data_path = Path(str(bainite_boundaries.PROJECT_ROOT), 'bainite_boundaries', 'bainite_boundaries', 'results', 'grid')
        
        # Bainite_start = np.load(results_path / f"Bainite_start_{model_Bainite}_y_pred_test.npy").mean(axis=0) #-273.15
        # Martensite_start = np.load(results_path / f"Martensite_start_{model_Martensite}_y_pred_test.npy").mean(axis=0) #-273.15
        
        # load csv files
        Bainite_start_df = pd.read_csv(data_path / f'Bainite_start_{model_Bainite}_grid_prediction.csv')
        Bainite_start = Bainite_start_df['y_pred'].to_numpy()
        Martensite_start_df = pd.read_csv(data_path / f'Martensite_start_{model_Martensite}_grid_prediction.csv')
        Martensite_start = Martensite_start_df['y_pred'].to_numpy()
        T_test = Bainite_start * alpha + Martensite_start * (1-alpha)

        x_test_final_padded[:, -1] = T_test
        T_train = np.concatenate(x)[:, -1]
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(Martensite_start, bins=50, alpha=0.5, label='Martensite start', density=True)
        ax.hist(Bainite_start, bins=50, alpha=0.5, label='Bainite start', density=True)
        ax.hist
        ax.legend()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.hist(T_test, bins=50, alpha=0.5, label='T_test', density=True)
        ax.hist(T_train, bins=50, alpha=0.5, label='T_train', density=True)
        ax.legend() 
        plt.show()

    
    return x_test_final_padded

def dict_to_filename(params):
    # Convert dictionary to a string with JSON
    params_str = json.dumps(params)
    
    # Replace characters that are not allowed in filenames
    invalid_chars = {':': '-', ',': '_', '{': '', '}': '', '\"': '', ' ': '', '\'': ''}
    for char, replacement in invalid_chars.items():
        params_str = params_str.replace(char, replacement)
    
    return params_str




# Function to extract the source code of a lambda function
def get_lambda_source(func):
    source_code = inspect.getsource(func).strip()  # Get the source code
    return source_code[source_code.find(":") + 1:].strip()  # Extract after colon


def get_monotonicity_constraints( which_data, features):
    if which_data in ['Bainite', 'Ferrite']:
        feature_monotonicity_dict = {
            'c': 1, 
            'si': 1,
            'mn': 1,
            'cr': 1,
            'v': 1,
            'co': -1,
            'al': -1,
            'n': 0,
            'b': 0,
            'p': 0,
            'ti': 0,
            's': 0,
            'w': 0,
            'cu': 0,
            'mo': 1,
            'ni': 1,
            'nb': 0,
            't' : 0,
            "T**2":0, 
            "log(abs(T) + 1)": 0, 
            "(log(abs(T) + 1))**2":0
        }
        selected_constraints = [feature_monotonicity_dict[feature] for feature in features]
        selected_constraints += [0, 0,0]
        monotone_constraints = tuple(selected_constraints)
    elif which_data in ['Bainite_start']:
        feature_monotonicity_dict = {
            'c': -1, 
            'si': -1,
            'mn': -1,
            'cr': -1,
            'v': -1,
            'co': -1,
            'al': 1,
            'mo': -1,
            'ni': -1,
        }
        monotone_constraints = [feature_monotonicity_dict[feature] for feature in features]
    else: 
        monotone_constraints = None

    return monotone_constraints




import numpy as np
import pandas as pd

def select_best_model_for_metrics(model_names, results_models, which_data, CP_params, metrics_type='accuracy'):
    """
    Function to select the best model for a given metrics type (accuracy or uncertainty).
    
    Parameters:
    - model_names: List of model names
    - results_models: Dictionary of model results
    - which_data: The dataset to select (Austensite, Martensite_start, etc.)
    - CP_params: Dictionary with conformal prediction parameters
    - metrics_type: The type of metrics to select (accuracy, uncertainty, or distance_measure)
    
    Returns:
    - best_models: Dictionary of best models for each selected metric
    """
    # Define the accuracy and uncertainty metrics based on which_data
    accuracy_metrics = {
        'Austensite': ['accuracy'], 
        'Martensite_start': ['MAPE'],
        'Martensite_start_RA': ['MAPE'],
        'Bainite_start': ['MAPE'],
        'Bainite': ['MALE'], 
        'Ferrite': ['MALE'], 
        'Ferrite_critCR': ['MAE']
    }[which_data]

    uncertainty_metrics = { 
        'Austensite': ['coverage', 'interval_size', 'coverage_distance', 'interval_size_distance'],
        'Martensite_start': ['coverage', 'interval_size', 'coverage_distance', 'interval_size_distance'],
        'Martensite_start_RA': ['coverage', 'interval_size', 'coverage_distance', 'interval_size_distance'],
        'Bainite_start': ['coverage', 'interval_size', 'coverage_distance', 'interval_size_distance'],
        'Bainite': ['coverage', 'interval_size', 'coverage_distance', 'interval_size_distance'],
        'Ferrite': ['coverage', 'interval_size', 'coverage_distance', 'interval_size_distance'],
        'Ferrite_critCR': ['coverage', 'interval_size', 'coverage_distance', 'interval_size_distance']
    }[which_data]
    
    distance_measure_metrics = {
        'Austensite': ['coverage_distance', 'interval_size_distance'],
        'Martensite_start': ['coverage_distance', 'interval_size_distance'],
        'Martensite_start_RA': ['coverage_distance', 'interval_size_distance'],
        'Bainite_start': ['coverage_distance', 'interval_size_distance'],
        'Bainite': ['coverage_distance', 'interval_size_distance'],
        'Ferrite': ['coverage_distance', 'interval_size_distance'],
        'Ferrite_critCR': ['coverage_distance', 'interval_size_distance']   
    }[which_data]
    

    # Select the metrics based on the metrics_type parameter
    if metrics_type == 'accuracy':
        metrics = accuracy_metrics
    elif metrics_type == 'uncertainty':
        metrics = uncertainty_metrics
    elif metrics_type == 'distance_measure':
        metrics = distance_measure_metrics
    else:
        raise ValueError("metrics_type not found")

    # Initialize a dictionary to store the results for each metric
    results_metrics = {metric: {} for metric in metrics}

    # Obtain the results for each model and metric
    for model_name_1 in model_names:
        results_df = results_models[model_name_1]
        for metric in metrics:
            metric_values = np.array([row[metric] for row in results_df['error_test']])
            results_metrics[metric][model_name_1] = np.nanmean(metric_values)

    # Define the selection criteria for each metric
    selection_criteria = {
        'MAPE': lambda x: min(x, key=x.get),
        'MAE': lambda x: min(x, key=x.get),
        'R2': lambda x: max(x, key=x.get),
        'MALE': lambda x: min(x, key=x.get),  
        'accuracy': lambda x: max(x, key=x.get),
        'NLL': lambda x: min(x, key=x.get),
        'coverage': lambda x: min(x, key=lambda k: abs(x[k] - CP_params['coverage']*100)),  
        'interval_size': lambda x: min(x, key=x.get),
        'coverage_distance': lambda x: min(x, key=lambda k: abs(x[k] - CP_params['coverage']*100)), 
        'interval_size_distance': lambda x: min(x, key=x.get),  
    }

    # Apply selection criteria and store the best model for each metric
    best_models = {}
    for metric in metrics:
        best_model = selection_criteria[metric](results_metrics[metric])
        best_models[metric] = best_model

    return best_models


def compute_metrics_classification(prediction_dict):

    y_true = prediction_dict['y_true']
    y_pred = prediction_dict['y_pred']
    y_pred_proba = prediction_dict['y_pred_proba']
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = np.mean(y_true == prediction_dict['y_pred']) * 100
    nll = log_loss(y_true, y_pred_proba[:, 1])
    brier = brier_score_loss(y_true, y_pred_proba[:, 1])
    f1 = f1_score(y_true, y_pred) * 100
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    coverage = np.mean( prediction_dict['confidence_sets'][np.arange(len(y_true)),y_true.astype(int)] )
    interval_size = np.mean( np.sum(prediction_dict['confidence_sets'], axis=1) )
    coverage_distance = np.mean( prediction_dict['confidence_sets_distance'][np.arange(len(y_true)),y_true.astype(int)] )
    interval_size_distance = np.mean( np.sum(prediction_dict['confidence_sets_distance'], axis=1) )
            
    error_dict = { 'accuracy': accuracy, 
                    'NLL': nll, 
                    'Brier': brier, 
                    'F1': f1, 
                    'ROC_AUC': roc_auc, 
                    'coverage': coverage, 
                    'interval_size': interval_size, 
                    'coverage_distance': coverage_distance,
                    'interval_size_distance': interval_size_distance,
                    'constraint_violation': constraint_violation
                    }

def compute_metrics_regression(prediction_dict):
    y_pred = prediction_dict['y_pred']
    y_true = prediction_dict['y_true']
                    
    MALE = mean_absolute_error( stable_log10( np.abs(y_true) + 1e-6), stable_log10( np.abs(y_pred) + 1e-6) )
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) 
    MAE = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    r2_log = 1 - np.sum((stable_log10(y_true) - stable_log10(y_pred))**2) / np.sum((stable_log10(y_true) - np.mean(stable_log10(y_true)))**2)
    
    coverage = np.mean( (y_true >= prediction_dict['lower_CP']) & (y_true <= prediction_dict['upper_CP']) ) 
    interval_size = np.mean( (prediction_dict['upper_CP'] - prediction_dict['lower_CP'])  )
    interval_log_size = np.mean( (stable_log10( np.abs(prediction_dict['upper_CP']) + 1e-6) - stable_log10( np.abs(prediction_dict['lower_CP']) + 1e-6) ) )
    
    coverage_distance = np.mean( (y_true >= prediction_dict['lower_CP_distance']) & (y_true <= prediction_dict['upper_CP_distance']) ) 
    interval_size_distance = np.mean( ( prediction_dict['upper_CP_distance'] - prediction_dict['lower_CP_distance'] )  )
    interval_log_size_distance = np.mean( (stable_log10( np.abs(prediction_dict['upper_CP_distance']) + 1e-6) - stable_log10( np.abs(prediction_dict['lower_CP_distance']) + 1e-6) ) )
        
    metrics_dict = {
        'MALE': MALE, 
        'MAPE [%]': MAPE*100,
        'MAE': MAE,
        'R2': r2,
        'R2log': r2_log,
        'coverage [%]': coverage*100,
        'coverage_distance [%]': coverage_distance*100,
        'interval_size ': interval_size,
        'interval_size_distance ': interval_size_distance,
        'interval_log_size': interval_log_size,
        'interval_log_size_distance': interval_log_size_distance
        
    }
    
    # # print key value pairs
    # for key, value in metrics_dict.items():
    #     print(key, value)
    
    return metrics_dict
    

def display_metrics(results_df, which_data, log=False):
    """
    Displays metrics for a given dataset, handling the specific metrics
    for 'Austensite' differently than for other datasets.
    
    Parameters:
    - results_df: Dictionary containing 'error_train', 'error_calib', 'error_test' data.
    - which_data: The dataset to display ('Austensite' or other options).
    """
    if which_data == "Austensite":
        metrics_list = ['accuracy', 'Brier', 'NLL', 'coverage', 'interval_size', 
                        'coverage_distance', 'interval_size_distance', 'constraint_violation']
 
    else:
        metrics_list = ['MALE', 'MAPE', 'MAE', 'R2', 'R2_log', 'coverage', 'interval_size', 
                        'coverage_distance', 'interval_size_distance', 
                        'coverage_orig', 'interval_size_orig']
        
        if log:
            metrics_list += ['interval_log_size', 'interval_log_size_distance', 'interval_log_size_orig']
        
    # Display error metrics for each type ('train', 'calib', 'test')
    for idx in ['error_train', 'error_calib', 'error_test']:
        print("*" * 10)
        for metric in metrics_list:
            if metric in results_df[idx][0].keys():
                metric_values = np.array([row[metric] for row in results_df[idx]])
                mean_value = np.nanmean(metric_values)
                std_value = np.nanstd(metric_values)
                print(f"{idx}: {metric} = {mean_value:.2f}% ± {std_value:.2f}%")
    

def stable_log10(x, epsilon=1e-6, max_value=1e8):
    # Clip values to be within the range [epsilon, max_value]
    x_clipped = np.clip(x, epsilon, max_value)
    return np.log10(x_clipped)