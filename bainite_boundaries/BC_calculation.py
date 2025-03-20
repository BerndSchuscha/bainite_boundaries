# %%
import os
import sys
from pathlib import Path
import bainite_boundaries
project_root = os.path.abspath(Path(str(bainite_boundaries.PROJECT_ROOT), 'bainite_boundaries', 'bainite_boundaries'))

# Add `bainite_boundaries` to sys.path if it’s not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    print("Ipython")
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
    show_plots = True

import numpy as np
import pandas as pd
from collections import Counter

import torch
import gpytorch
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


from models.GaussianProcess import ExactGPModel
from models.XGBoost import XGBRegressorModel
from models.RandomForest import RandomForestRegressorModel
from models.LGBM import LGBMRegressorModel
from models.linear import LinearRegressionModel, LinearClassificationModel
from models.PolynomialRegression import PolynomialRegressionModel
from models.RandomFeatures import RandomFeatureModel
from models.NeuralNetwork import NeuralNetworkModel
from bainite_boundaries.models.Monoton_NN import Monoton_NeuralNetworkModelPytorch

from models.GaussianProcess_Poly import ExactGPModel_Poly
from models.GaussianProcess_linearmean import ExactGPModel_linearmean
from models.NeuralNetwork_pytorch import NeuralNetworkModelPytorch
from models.PolynomialRegression import PolynomialRegressionModel_mixeddata
from models.linear import LinearRegressionModel_mixeddata

from utils.data_processing import extend_with_constraints_final, get_monotonicity_constraints, select_best_model_for_metrics, display_metrics
from utils.plotting import plot_kfold_results, plot_predictions
from utils.load_data import load_dataset
from utils.train import run_cross_validation



#%% Load the data
print(os.getcwd())
which_data = 'Austensite'
#which_data = 'Martensite_start'
which_data = 'Bainite_start'
which_data = 'Bainite'

#which_data = 'Ferrite'
#which_data = 'Ferrite_critCR'
x, y, features = load_dataset(which_data)
x_test_final, y_test_austensite, features_final = load_dataset('final')
print("Features: ", features)
print("Features austensite: ", features_final)
x_test_final_padded = extend_with_constraints_final(x_test_final, features, features_final)


# %% Define the model
 
# regression: LGBM RF GP XGB XGB_no_monotonic LGBM_no_monotonic linear polynomial random
# classification: RF_class GB_class SVC_class MLP_class LR_class linear_class

model_names = {
    'Austensite': ['RF_class'],#, 'GB_class', 'SVC_class'],
    'Bainite': ['linear'],
    # 'Bainite': ['GP'],
    'Ferrite': ['linear', 'RF', 'XGB_no_monotonic', 'LGBM_no_monotonic','polynomial'],#['linear', 'neural', 'neural_pt'],#['linear', 'neural_pt','neural_pt'],
    'Martensite_start':['linear'],# ['monoton_neural','linear','polynomial','neural','neural_pt','random','GP','GP_Poly'],#,'GP_Poly'],
    'Bainite_start': ['linear'],# 'RF', 'XGB_no_monotonic', 'LGBM_no_monotonic'],
    'Ferrite_critCR': ['RF']#['polynomial','XGB_no_monotonic','LGBM_no_monotonic', 'GP','GP_Poly','NN']    
}[which_data]

if which_data == 'Austensite':
    classification = True
else:
    classification = False

distance_metric_locs = [
                        ('PCA_scaled', 'knn'), 
                        #('PCA_scaled', 'likelihood_ratio'),
                        #('PCA_scaled', 'mixture_model'),
                        #('PCA_scaled', 'lof'),
                        ('PCA_scaled', 'mahalanobis'),
                        #('PCA_scaled', 'wasserstein'),
                        # ('PCA_scaled', 'wasserstein_truncated'),
                        #('PCA_scaled', 'cosine'),
                        #('PCA_scaled', 'jensen_shannon'),
                        #('PCA_scaled', 'energy'),
]
 
CP_params = {
    'coverage': 0.9,
    'distance_scale': 5,
    'distance_metric': 'wasserstein_truncated', # knn, likelihood_ratio, mixture_model, lof, mahalanobis, wasserstein, wasserstein_truncated
    'distance_loc': 'PCA_scaled',  # PCA, PCA_scaled, input, input_scaled
    'n_neighbors': 5
}

if which_data in ['Bainite','Ferrite']:

    transformations = {
        "T**2": lambda T: T**2, 
        # "T**4": lambda T: T**4, 
        "log(abs(T) + 1)": lambda T: np.log(np.abs(T) + 1), 
        "(log(abs(T) + 1))**2": lambda T: (np.log(np.abs(T) + 1))**2, 
        # "(log(abs(T) + 1))**3": lambda T: (np.log(np.abs(T) + 1))**3,
        # "(log(abs(T) + 1))**4": lambda T: (np.log(np.abs(T) + 1))**4,
    }
else:
    transformations=None

random_state = np.random.randint(0, 1000)
random_state_val = np.random.randint(0, 1000)
    
input_transform = True
if which_data in ['Bainite', 'Ferrite', 'Ferrite_critCR']:
    output_transform = 'log'
else:
    output_transform = None


monotone_constraints = get_monotonicity_constraints(which_data, features)
model_parameters = {
    'linear': {
        'apply_pca': False,
        'explained_variance': 0.95, 
        'transformations': transformations, 
        'alpha': 0.0001
    },
    'polynomial': {
        'apply_pca': True,
        'explained_variance': 0.95,
        'degree': 2
    },
    'neural': {
        'hidden_size': 200,
        'transformations': transformations
    },
    'mono_neural': {
        'hidden_size': 200,
        'transformations': transformations,
        'monotonicty': monotone_constraints
    },
    'random': {
        'n_components': 100,
        'apply_pca': False,
        'explained_variance': 0.9
    },
    'RF': {
        'n_estimators': 200,
        'max_depth': 50,
        'random_state': random_state, 
        'transformations': transformations
    },
    'XGB': {
        'n_estimators': 10,
        'max_depth': 10,
        'random_state': random_state,
        'monotone_constraints': monotone_constraints
    },
    'LGBM': {
        'n_estimators': 10,
        'max_depth': 10,
        'random_state': random_state,
        'monotone_constraints': monotone_constraints
    },
    'XGB_no_monotonic': {
        'n_estimators': 10,
        'max_depth': 10,
        'random_state': random_state, 
    },
    'LGBM_no_monotonic': {
        'n_estimators': 10,
        'max_depth': 10,
        'random_state': random_state
    },
    'GP': {'num_epochs': 2000,
        'lr': 0.01,
        'range_scale': 1.0
    }
}

model_dict = {
    'linear': LinearRegressionModel(**model_parameters['linear']),
    'polynomial': PolynomialRegressionModel(**model_parameters['polynomial']),
    
    'neural': NeuralNetworkModel(**model_parameters['neural']),
    'neural_pt': NeuralNetworkModelPytorch(**model_parameters['neural']),
    'monoton_neural': Monoton_NeuralNetworkModelPytorch(**model_parameters['mono_neural']),
    'random': RandomFeatureModel(**model_parameters['random']),
    'RF': RandomForestRegressorModel(**model_parameters['RF']),
    'XGB': XGBRegressorModel(**model_parameters['XGB']),
    'LGBM': LGBMRegressorModel(**model_parameters['LGBM']),
    'XGB_no_monotonic': XGBRegressorModel(**model_parameters['XGB_no_monotonic']),
    'LGBM_no_monotonic': LGBMRegressorModel(**model_parameters['LGBM_no_monotonic']),
    'GP': ExactGPModel(torch.tensor(x[0]).float().to(device), 
                       torch.tensor(y[0]).float().to(device), 
                       gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-1),), 
                       **model_parameters['GP']),
    'GP_Poly': ExactGPModel_Poly(torch.tensor(x[0]).float().to(device), 
                       torch.tensor(y[0]).float().to(device), 
                       gpytorch.likelihoods.GaussianLikelihood(), 
                       **model_parameters['GP']),
    
    'GP_linearmean': ExactGPModel_linearmean(torch.tensor(x[0]).float().to(device), 
                    torch.tensor(y[0]).float().to(device), 
                    gpytorch.likelihoods.GaussianLikelihood(), 
                    **model_parameters['GP']),
    # Implimentation for mixed data
    'linear_mixed': LinearRegressionModel_mixeddata(**model_parameters['linear']),
    'polynomial_mixed': PolynomialRegressionModel_mixeddata(**model_parameters['polynomial']),

    # classification 
    'RF_class': RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42),
    'GB_class': GradientBoostingClassifier(n_estimators=5, max_depth=10, random_state=42),
    'SVC_class': SVC(kernel='rbf', C=100000, gamma='scale'),
    'MLP_class': MLPClassifier(hidden_layer_sizes=(10000), random_state=42, max_iter=500, activation='relu', solver='adam'),
    'LR_class': LogisticRegression(max_iter=1000),
    'linear_class': LinearClassificationModel()
}



# %% k fold cross validation to evaluate the model
# number of splits for the cross validation for conformal prediction
n_splits_CP = 5

n_splits = 10
x_test, y_test = None, None
results_models = {}
for model_name_1 in model_names:
    print(f"Model: {model_name_1}")
    results_distance_metrics = {}
    distance_metrics = [item[1] for item in distance_metric_locs]
    
    for distance_loc, distance_metric in distance_metric_locs:
        print(f"    Distance metric: {distance_metric}, Distance loc: {distance_loc}")
        CP_params['distance_loc'] = distance_loc
        CP_params['distance_metric'] = distance_metric
        results_df = run_cross_validation(x, y, 
                                        x_test_final=x_test,
                                        y_test_final=y_test,
                                        model=model_dict[model_name_1],
                                        CP_params=CP_params,
                                        n_splits=n_splits, n_splits_CP=n_splits_CP, 
                                        input_transform=input_transform, output_transform=output_transform, 
                                        classification=classification,
                                        verbose=0, show_plots=True)
        results_distance_metrics[distance_metric] = results_df
    print(results_distance_metrics)
    best_models_uncertainty = select_best_model_for_metrics(distance_metrics, results_distance_metrics, which_data, CP_params, metrics_type='distance_measure')
    best_uncertainty = Counter(best_models_uncertainty).most_common(1)[0][1]
    best_input = distance_metric_locs[distance_metrics.index(best_uncertainty)][0]
    print(f"Best model for uncertainty: {best_uncertainty}")
    
    results_df = results_distance_metrics[best_uncertainty]
    results_models[model_name_1] = results_df
    
    results_path = Path(str(bainite_boundaries.PROJECT_ROOT), 'bainite_boundaries', 'bainite_boundaries', 'results', 'k-fold-CV')
    results_name = f"{which_data}_{model_name_1}_results"
    results_df.drop(columns=['pipeline']).to_csv(results_path/f'{results_name}.csv', index=False)



#%%
#  Select the best model based on the k fold cross validation results
best_models_accuracy = select_best_model_for_metrics(model_names, results_models, which_data, CP_params, metrics_type='accuracy')

best_model = Counter(best_models_accuracy).most_common(1)[0][1]
results_df = results_models[best_model]
print(f"Best model for accuracy {best_model}, and uncertainty {results_df['distance_metric'][0]}")

# %% Plot the results of the k fold cross validation
display_metrics(results_df, which_data=which_data, log=output_transform=='log')
plot_kfold_results(results_df, which_data=which_data, model_name_1=best_model, log=output_transform=='log')

grid=True
#%% Final evaluation on the test set  
if grid:
    n_splits = 1

    best_uncertainty = results_df['distance_metric'][0]
    best_input = results_df['distance_loc'][0]
    print(f"Model: {best_model}, Distance metric: {best_uncertainty}, Distance loc: {best_input}")
    CP_params['distance_loc'] = best_input
    CP_params['distance_metric'] = best_uncertainty

    x_test, y_test = x_test_final_padded, None
    
    results_df = run_cross_validation(x, y, 
                                    x_test_final=x_test,
                                    y_test_final=y_test,
                                    model=model_dict[best_model],
                                    CP_params=CP_params,
                                    n_splits=n_splits, n_splits_CP=n_splits_CP, 
                                    input_transform=input_transform, output_transform=output_transform, 
                                    classification=classification,
                                    verbose=2, show_plots=False,save=True,which_data=which_data)
    display_metrics(results_df, which_data=which_data, log=output_transform=='log')


#%% Prediction on the test set from a saved model
if grid:
    y_pred_test = []
    upper_CP_test, lower_CP_test = [], []
    upper_CP_distance_test, lower_CP_distance_test = [], []


    for idx in range(n_splits_CP):
        pipeline = results_df['pipeline'][idx]
        distance_test = pipeline.calculate_distances.distance_transform(x_test_final_padded)
        prediction_dict_test = pipeline.get_interval_predictions(x_test_final_padded, CP_params['coverage'], distance=distance_test)
        y_pred_test.append(prediction_dict_test['y_pred'])
        
        if which_data != 'Austensite':
            upper_CP_test.append(prediction_dict_test['upper_CP'])
            lower_CP_test.append(prediction_dict_test['lower_CP'])
            upper_CP_distance_test.append(prediction_dict_test['upper_CP_distance'])
            lower_CP_distance_test.append(prediction_dict_test['lower_CP_distance'])
            
        
    y_pred_test = np.array(y_pred_test)
    y_pred_test_mean = np.mean(y_pred_test, axis=0)
    y_pred_test_std = np.std(y_pred_test, axis=0)

    upper_CP_test = np.array(upper_CP_test).mean(axis=0)
    lower_CP_test = np.array(lower_CP_test).mean(axis=0)
    upper_CP_distance_test = np.array(upper_CP_distance_test).mean(axis=0)
    lower_CP_distance_test = np.array(lower_CP_distance_test).mean(axis=0)

    grid_prediction_dict = {
                'y_pred': y_pred_test_mean,
                'lower_CP': lower_CP_test,
                'upper_CP': upper_CP_test,
                'lower_CP_distance': lower_CP_distance_test,
                'upper_CP_distance': upper_CP_distance_test,
            }

    # save to results folder
    results_path = Path(str(bainite_boundaries.PROJECT_ROOT), 'bainite_boundaries', 'bainite_boundaries', 'results', 'grid')
    # save grid prediction dict as csv
    grid_prediction_df = pd.DataFrame(grid_prediction_dict)
    grid_prediction_df.to_csv(results_path/f'{which_data}_grid_prediction.csv', index=False)


    # Plot the predictions on the final grid

    plt.figure()
    plt.plot(y_pred_test_mean, label='Prediction')
    plt.fill_between(np.arange(len(y_pred_test_mean)), y_pred_test_mean - y_pred_test_std, y_pred_test_mean + y_pred_test_std, alpha=0.4)
    if which_data != 'Austensite':
        plt.fill_between(np.arange(len(y_pred_test_mean)), lower_CP_test, upper_CP_test, alpha=0.4, color='green', label='CP')
        plt.fill_between(np.arange(len(y_pred_test_mean)), lower_CP_distance_test, upper_CP_distance_test, alpha=0.4, color='red', label='CP distance')
    # plt.xlim(0,100)
    plt.legend()
    if output_transform == 'log':
        plt.yscale('log')
    plt.show()

    plt.figure(figsize=(2.5, 2.5))
    plt.plot(lower_CP_distance_test-lower_CP_test, label='Lower difference')
    plt.plot(upper_CP_distance_test-upper_CP_test, label='Upper difference')
    plt.legend()
    plt.show()

#%% Constraint violation
if grid:
    if which_data == 'Bainite':
        constraint_violation = np.mean(y_pred_test_mean > 3*60*60) * 100
        print(f'Constraint violation: {constraint_violation:.2f}%')
        constraint_violation_lower = np.mean(lower_CP_test > 3*60*60) * 100
        print(f'Constraint violation lower: {constraint_violation_lower:.2f}%')
        constraint_violation_upper = np.mean(upper_CP_test > 3*60*60) * 100
        print(f'Constraint violation upper: {constraint_violation_upper:.2f}%')
        constraint_violation_lower_distance = np.mean(lower_CP_distance_test > 3*60*60) * 100
        print(f'Constraint violation lower distance: {constraint_violation_lower_distance:.2f}%')
        constraint_violation_upper_distance = np.mean(upper_CP_distance_test > 3*60*60) * 100
        print(f'Constraint violation upper distance: {constraint_violation_upper_distance:.2f}%')
        



    # %%