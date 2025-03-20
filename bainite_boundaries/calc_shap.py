# %%
import shap
import time
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
from utils.load_data import load_dataset
from utils.train import run_cross_validation
from bainite_boundaries.utils.data_processing import evaluation_conditions

def ensemble_predict(X, results_df, which_data,classifiy=False):
    if classifiy:
        X_process=X[:,-2:]
        X=X[:,:-2]

    preds = []
    n_splits = len(results_df['CP_CV_fold'].unique())
    
    for idx in range(n_splits):  # Assuming you have 3 models
        pipeline = results_df['pipeline'][idx]  # Get model from pipeline
        model = pipeline.model
        scaler = pipeline.scaler  # Feature scaling (if used)
        distance = pipeline.calculate_distances.distance_transform(X)
        
        if scaler is not None:
            X_scaled = scaler.transform(X)  # Apply the same transformation
        else:
            X_scaled = X
        prediction = model.predict(X_scaled)
        prediction_dict = pipeline.get_interval_predictions(X, coverage=.9, distance=distance)
        
        if classifiy:
            if which_data == 'Austensite':
                prediction = prediction_dict['y_pred']
            elif which_data in ['Martensite_start', 'Martensite_start_RA','Bainite', 'Ferrite_critCR']:
                prediction = prediction_dict['lower_CP_distance']
            elif which_data == 'Bainite_start':  
                prediction = prediction_dict['upper_CP_distance']
            else:
                prediction = prediction_dict['y_pred']

        if pipeline.output_transform == 'log' and not classifiy:
            # preds.append(np.exp( model.predict(X_scaled)) )
            prediction = np.log10(np.exp(prediction))
        preds.append(prediction)

    y=np.mean(preds, axis=0)
    if classifiy:
        y=evaluation_conditions(y,X_process,which_data)
        y=np.logical_not(y)
    return y  







#%% Load the data
print(os.getcwd())
which_data = 'Austensite'
# which_data = 'Martensite_start'
which_data = 'Bainite_start'
# which_data = 'Bainite'
# which_data = 'Martensite_start_RA'
# which_data = 'Ferrite_critCR'

which_data_list=['Bainite', 'Ferrite_critCR']

for which_data in which_data_list:
    x, y, features = load_dataset(which_data)
    x_test_final, y_test_austensite, features_final = load_dataset('final')
    print("Features: ", features)
    print("Features austensite: ", features_final)
    x_test_final_padded = extend_with_constraints_final(x_test_final, features, features_final)

    x_test_final_padded=np.append(x_test_final_padded,x_test_final[:,-2:],axis=1)
    # %% Define the model
    
    # regression: LGBM RF GP XGB XGB_no_monotonic LGBM_no_monotonic linear polynomial random
    # classification: RF_class GB_class SVC_class MLP_class LR_class linear_class




    model_names = {
        'Austensite': ['MLP_class'],
        'Bainite_start': ['GP_linearmean'],
        'Martensite_start':['GP_linearmean'],
        'Bainite': ['linear'],
        'Martensite_start_RA': ['GP_linearmean'],
        'Ferrite_critCR': ['RF'], 
    }[which_data]

    if which_data == 'Austensite':
        classification = True
        #x = x[::2000, :]
        #y = y[::2000]
    else:
        classification = False

    distance_metric_locs = [
                            ('PCA_scaled', 'knn'), 
                            # ('PCA_scaled', 'likelihood_ratio'),
                            # ('PCA_scaled', 'mixture_model'),
                            # ('PCA_scaled', 'lof'),
                            # ('PCA_scaled', 'mahalanobis'),
                            # ('PCA_scaled', 'wasserstein'),
                            # ('PCA_scaled', 'wasserstein_truncated'),
                            # ('PCA_scaled', 'cosine'),
                            # ('PCA_scaled', 'jensen_shannon'),
                            # ('PCA_scaled', 'energy'),
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
            # "log(abs(T) + 1)": lambda T: np.log(np.abs(T) + 1), 
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
            'alpha': 0.001, 
            # 'search_alphas': [0.1, 1, 10]
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
        'MLP_class': MLPClassifier(hidden_layer_sizes=(100), random_state=42, max_iter=500, activation='relu', solver='adam'),
        'LR_class': LogisticRegression(max_iter=1000),
        'linear_class': LinearClassificationModel()
    }



    # %% k fold cross validation to evaluate the model
    # number of splits for the cross validation for conformal prediction
    n_splits_CP = 5
    n_splits = 1
    x_test, y_test = None,None
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
                                            verbose=0, show_plots=False)
            results_distance_metrics[distance_metric] = results_df

    if which_data in ['Bainite']:
        X_total =  np.concatenate(x)
    else:
        X_total = np.array(x)




    # Convert the numpy array to a DataFrame
    X_test_final_padded_df = pd.DataFrame(x_test_final_padded)
    X_test_final = X_test_final_padded_df.sample(n=2000, random_state=42).to_numpy()

    if which_data in ['Bainite']:
        X_total =  np.concatenate(x)
    else:
        X_total = np.array(x)



    import shap
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from matplotlib import colormaps

    if len(X_total) > 1000:
        print("Reducing the training set to 200 samples")
        X_total = X_total[np.random.choice(X_total.shape[0], 1000, replace=False)]
    # 1. Reduce training set to 10 representative samples using k-means
    X_train_summary = shap.kmeans(X_total, int(np.min([np.shape(X_total)[0],100])))  # Reduce training set to 10 representative samples
    # 2. Create the SHAP explainer using the ensemble prediction function
    explainer = shap.KernelExplainer(lambda X: ensemble_predict(X, results_df, which_data), X_train_summary)



    # 3. Compute SHAP values for X_test_final and X_total
    shap_values_train = explainer.shap_values(X_total)     # Compute SHAP values for X_total

    # 4. Prepare the output directory for results
    results_path = Path(str(bainite_boundaries.PROJECT_ROOT), 'bainite_boundaries', 'bainite_boundaries', 'results', 'shap')

    # 5. Create DataFrames for SHAP values and features
    shap_values_df_train = pd.DataFrame(shap_values_train, columns=[f"SHAP_{feature}" for feature in features])

    # Convert and X_total to DataFrames if they're NumPy arrays
    X_total_df = pd.DataFrame(X_total, columns=[f"X_{feature}" for feature in features])

    # 6. Optionally, apply exponential transformation to SHAP values for specific datasets
    if which_data in ['Bainite', 'Ferrite']:

        shap_values_train = np.exp(shap_values_train)

    # 7. Combine the original features and SHAP values for both datasets
    combined_df_train = pd.concat([X_total_df, shap_values_df_train], axis=1)

    # 8. Save the data to CSV files
    combined_df_train.to_csv(results_path/f"{which_data}_train.csv", index=False)

    # 9. Create custom color map for the SHAP plots
    cmap = colormaps.get_cmap('coolwarm')


    # 11. Create the SHAP summary plot for X_total (training data)

    features=features+['T',r'$\dot{T}$']


    if len(X_test_final) > 1000:
        print("Reducing the training set to 1000 samples")
        X_test_final = X_test_final[np.random.choice(X_test_final.shape[0], 1000, replace=False)]
    # 1. Reduce training set to 10 representative samples using k-means
    X_train_summary = shap.kmeans(X_test_final, 100)  # Reduce training set to 10 representative samples
    # 2. Create the SHAP explainer using the ensemble prediction function
    explainer = shap.KernelExplainer(lambda X: ensemble_predict(X, results_df, which_data,classifiy=True), X_train_summary)



    # 3. Compute SHAP values for X_test_final and X_total
    shap_values_train = explainer.shap_values(X_test_final)     # Compute SHAP values for X_total

    # 4. Prepare the output directory for results
    results_path = Path(str(bainite_boundaries.PROJECT_ROOT), 'bainite_boundaries', 'bainite_boundaries', 'results', 'shap')

    # 5. Create DataFrames for SHAP values and features
    shap_values_df_train = pd.DataFrame(shap_values_train, columns=[f"SHAP_{feature}" for feature in features])

    # Convert X_test_final and X_total to DataFrames if they're NumPy arrays
    X_test_final_df = pd.DataFrame(X_test_final, columns=[f"X_{feature}" for feature in features])

    # 6. Optionally, apply exponential transformation to SHAP values for specific datasets
    if which_data in ['Bainite', 'Ferrite']:

        shap_values_train = np.exp(shap_values_train)

    # 7. Combine the original features and SHAP values for both datasets
    combined_df_train = pd.concat([X_test_final_df, shap_values_df_train], axis=1)

    # 8. Save the data to CSV files
    combined_df_train.to_csv(results_path/f"{which_data}_grid.csv", index=False)

    # 9. Create custom color map for the SHAP plots
    cmap = colormaps.get_cmap('coolwarm')


    # 11. Create the SHAP summary plot for X_total (training data)

