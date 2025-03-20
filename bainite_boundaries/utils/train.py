import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# compute ece, mce, nll, auroc
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss, f1_score
from sklearn.model_selection import StratifiedKFold

from utils.dimensionality_reduction import  CalculateDistances
from utils.plotting import plot_true_vs_pred, plot_predictions, plot_confusion_matrix, plot_calibration_curve
from utils.data_processing import stable_log10
import json
from pathlib import Path
import bainite_boundaries
from sklearn.preprocessing import KBinsDiscretizer
import copy

class ModelPipeline:
    def __init__(self, model, model_name, input_transform=True, output_transform=None):
        self.model = model
        self.model_name = model_name
        self.input_transform = input_transform
        self.output_transform = output_transform
        # self.scaler = StandardScaler() if input_transform else None
        # robust scaler
        self.scaler = StandardScaler() if input_transform else None
        # self.scaler = MinMaxScaler() if input_transform else None

    def split_data(self, x, y, train_index, val_index=None, test_index=None, val_split=0.2, random_state=None):
        np.random.seed(random_state)
        
        if val_index is None:
            val_size = int(len(train_index) * val_split)
            val_index = np.random.choice(train_index, size=val_size, replace=False)
            train_proper_index = list(set(train_index) - set(val_index))
        else:
            train_proper_index = train_index

        if test_index is None:
            test_index = []
        else:
            test_index = test_index

        return train_proper_index, val_index, test_index

    def prepare_data(self, x, y, train_index, val_index=None, test_index=None, val_split=0.2, random_state=None):
        # Split train, validation, and test indices
        train_proper_index, val_index, test_index = self.split_data(x, y, train_index, val_index, test_index, val_split, random_state)

        # Prepare the data
        # if test index is empty, X_test and y_test will be None
        if len(test_index) == 0:
            X_test = None
            y_test = None
        else:
            X_test = [x[i] for i in test_index]
            y_test = [y[i] for i in test_index]

        
        X_train_proper = [x[i] for i in train_proper_index]
        y_train_proper = [y[i] for i in train_proper_index]
                
        X_calib = [x[i] for i in val_index]
        y_calib = [y[i] for i in val_index]
        
        # Concatenate arrays
        
        if isinstance(x, np.ndarray):
            
            X_train_proper = np.stack(X_train_proper, axis=0)
            y_train_proper = np.array(y_train_proper)
            
            X_calib = np.stack(X_calib, axis=0)
            y_calib = np.array(y_calib)
            
            X_train = np.concatenate((X_train_proper, X_calib), axis=0)
            y_train = np.concatenate((y_train_proper, y_calib), axis=0)
            
            X_test = np.stack(X_test, axis=0) if X_test else None
            y_test = np.array(y_test) if y_test else None
            
        else:
            
            X_train_proper = np.concatenate(X_train_proper, axis=0)
            y_train_proper = np.concatenate(y_train_proper, axis=0)
        
        
            X_calib = np.concatenate(X_calib, axis=0)
            y_calib = np.concatenate(y_calib, axis=0)
            
            X_train = np.concatenate((X_train_proper, X_calib), axis=0)
            y_train = np.concatenate((y_train_proper, y_calib), axis=0)
            
            X_test = np.concatenate(X_test, axis=0) if X_test else None
            y_test = np.concatenate(y_test, axis=0) if y_test else None

        
        return (X_train, y_train, X_calib, y_calib, X_train_proper, y_train_proper, 
                X_test, y_test)

    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x) if self.input_transform else x
    
    def fit(self, x, y):
        if self.input_transform:
            x = self.scaler.fit_transform(x)
        if self.output_transform == 'log':
            if np.any(y < 0):
                raise ValueError("Negative values in y")
            y = np.log( np.clip(y, 1e-6, 1e10) )
        self.model.fit(x, y)

        
    def quantile_scores(self, x, y, coverage, distance=None, distance_scale=1):
        
        if self.input_transform:
            x = self.scaler.transform(x)
            
        if self.output_transform == 'log':
            y = np.log( np.clip(y, 1e-6, 1e10) )
        
        if self.task_type == 'classification':
            y_pred_proba = self.model.predict_proba(x)
            true_class_indices = np.arange(len(y)), y.astype(int)
            nonconformity_scores = 1 - y_pred_proba[true_class_indices]
        else:
            y_pred = self.model.predict(x)
            nonconformity_scores = np.abs(y - y_pred)    
            
            if hasattr(self.model, "predict_uncertainty"):
                print('Using model uncertainty')
                model_uncertainty = self.model.predict_uncertainty(x)
                nonconformity_scores /= (1 + model_uncertainty)
             
        n = len(nonconformity_scores)
        self.CP_quantiles = np.quantile(nonconformity_scores, np.minimum(coverage * (n + 1) / n, 1))
        
        if distance is not None: 
            self.distance_scale = distance_scale
            self.distance_mean = distance.mean()
            distance_transformed = self.distance_scale + distance / self.distance_mean
            self.CP_quantiles_distance = np.quantile(nonconformity_scores / distance_transformed, np.minimum(coverage * (n + 1) / n, 1))
            
    
    def get_interval_predictions(self, x, coverage, distance=None):
        if self.input_transform:
            x = self.scaler.transform(x)
       
        if self.task_type == 'classification': 
            
            y_pred = self.model.predict(x)
            y_pred_proba = self.model.predict_proba(x)
            confidence_sets = (y_pred_proba >= 1 - self.CP_quantiles).astype(int)
            
            distance_scaled = self.distance_scale + distance / self.distance_mean
            distance_scaled = np.repeat(distance_scaled[:, np.newaxis], y_pred_proba.shape[1], axis=1)
            confidence_sets_distance = (y_pred_proba >= 1 - self.CP_quantiles_distance * distance_scaled).astype(int)
            
            prediction_dict = { 'y_pred': y_pred,
                                'y_pred_proba': y_pred_proba,
                                'confidence_sets': confidence_sets,
                                'confidence_sets_distance': confidence_sets_distance }
            
            return prediction_dict
            
        else:
            y_pred = self.model.predict(x)
            # Include model's uncertainty heuristic if available
            if hasattr(self.model, "predict_uncertainty"):
                model_uncertainty = self.model.predict_uncertainty(x)
            else:
                model_uncertainty = 0
            
            upper_CP = y_pred + self.CP_quantiles * (1 + model_uncertainty)
            lower_CP = y_pred - self.CP_quantiles * (1 + model_uncertainty)
            
       
            if distance is not None:
                distance_scaled = self.distance_scale + distance / self.distance_mean
                upper_CP_distance = y_pred + self.CP_quantiles_distance * distance_scaled * (1 + model_uncertainty)
                lower_CP_distance = y_pred - self.CP_quantiles_distance * distance_scaled * (1 + model_uncertainty)
            else: 
                upper_CP_distance = upper_CP
                lower_CP_distance = lower_CP
                    
            if self.output_transform == 'log':
                y_pred = np.exp(y_pred)
                upper_CP = np.exp(upper_CP)
                lower_CP = np.exp(lower_CP) 
                upper_CP_distance = np.exp(upper_CP_distance)
                lower_CP_distance = np.exp(lower_CP_distance)  
  
            
            prediction_dict = { 'y_pred': y_pred,
                                'upper_CP': upper_CP,
                                'lower_CP': lower_CP,
                                'upper_CP_distance': upper_CP_distance,
                                'lower_CP_distance': lower_CP_distance }

            return prediction_dict

    
    def compute_errors(self, y_true, prediction_dict):
        
        if self.task_type == 'classification':
            y_pred_proba = prediction_dict['y_pred_proba']
            y_pred = prediction_dict['y_pred']
            
            # constraint violation
            constraint_violation = np.mean( prediction_dict['confidence_sets'][:, 0] )
            
            if y_true is None:
                accuracy, nll, brier, f1, roc_auc = -1, -1, -1, -1, -1
                coverage, interval_size = -1, -1
                coverage_distance, interval_size_distance = -1, -1
            else:
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
        
        else:
            y_pred = prediction_dict['y_pred']
            threshold = 3600 * 3
            constraint_violation_pred = np.mean( y_pred > threshold ) * 100
            constraint_violation_lower = np.mean( prediction_dict['lower_CP'] > threshold ) * 100
            constraint_violation_upper = np.mean( prediction_dict['upper_CP'] > threshold ) * 100
            constraint_violation_lower_distance = np.mean( prediction_dict['lower_CP_distance'] > threshold ) * 100
            constraint_violation_upper_distance = np.mean( prediction_dict['upper_CP_distance'] > threshold ) * 100
            
            error_dict = {'constraint_violation_pred': constraint_violation_pred,
                        'constraint_violation_lower': constraint_violation_lower,
                        'constraint_violation_upper': constraint_violation_upper,
                        'constraint_violation_lower_distance': constraint_violation_lower_distance,
                        'constraint_violation_upper_distance': constraint_violation_upper_distance}
            
            if y_true is None:
                MALE, MAPE, MAE, r2 = -1, -1, -1, -1
                coverage, interval_size = -1, -1
                coverage_distance, interval_size_distance = -1, -1
                interval_log_size, interval_log_size_distance = -1, -1
            else:
                y_pred = prediction_dict['y_pred']
                
                MALE = mean_absolute_error( stable_log10( np.abs(y_true) + 1e-6), stable_log10( np.abs(y_pred) + 1e-6) )
                MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                MAE = np.mean(np.abs(y_true - y_pred))
                r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
                #r2_log = 1 - np.sum((stable_log10( np.abs(y_true) + 1e-6) - stable_log10( np.abs(y_pred) + 1e-6))**2) / np.sum((stable_log10( np.abs(y_true) + 1e-6) - np.mean(stable_log10( np.abs(y_true) + 1e-6)))**2)
                
                
                #coverage_orig = np.mean( (y_true >= prediction_dict['lower']) & (y_true <= prediction_dict['upper']) ) * 100
                coverage = np.mean( (y_true >= prediction_dict['lower_CP']) & (y_true <= prediction_dict['upper_CP']) ) * 100
                interval_size = np.mean( (prediction_dict['upper_CP'] - prediction_dict['lower_CP'])  )
                interval_log_size = np.mean( (stable_log10( np.abs(prediction_dict['upper_CP']) + 1e-6) - stable_log10( np.abs(prediction_dict['lower_CP']) + 1e-6) ) )
                
                coverage_distance = np.mean( (y_true >= prediction_dict['lower_CP_distance']) & (y_true <= prediction_dict['upper_CP_distance']) ) * 100
                interval_size_distance = np.mean( ( prediction_dict['upper_CP_distance'] - prediction_dict['lower_CP_distance'] ) )
                interval_log_size_distance = np.mean( (stable_log10( np.abs(prediction_dict['upper_CP_distance']) + 1e-6) - stable_log10( np.abs(prediction_dict['lower_CP_distance']) + 1e-6) ) )
                #interval_size_orig = np.mean( (prediction_dict['upper'] - prediction_dict['lower'])  )
                #interval_log_size_orig = np.mean( (stable_log10( np.abs(prediction_dict['upper']) + 1e-6) - stable_log10( np.abs(prediction_dict['lower']) + 1e-6) ) )
                
            error_dict = {  'MALE': MALE, 
                            'MAPE': MAPE, 
                            'MAE': MAE,
                            'R2': r2,
                            #'R2_log': r2_log,
                             #'coverage_orig': coverage_orig,
                            'coverage': coverage,
                            'interval_size': interval_size,
                            'interval_log_size': interval_log_size,
                            'coverage_distance': coverage_distance,
                            'interval_size_distance': interval_size_distance,
                            'interval_log_size_distance': interval_log_size_distance,
                            #'interval_size_orig': interval_size_orig,
                            #'interval_log_size_orig': interval_log_size_orig,
                            **error_dict
                            }
        return error_dict
    
    

        
        
def setup_logging(verbose):
    # Always disable the root logger to prevent duplicate logs
    logging.getLogger().disabled = True
    
    logger = logging.getLogger("MyCustomLogger")
    
    if verbose:
        logger.setLevel(logging.INFO)  # Set logging level to INFO
        
        # Check if any handlers are already present to avoid duplication
        if not logger.hasHandlers():
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Define the logging format (without INFO:root)
            formatter = logging.Formatter('%(message)s')
            ch.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(ch)
    else:
        logger.setLevel(logging.WARNING)  # Set logging level to WARNING
    
    return logger


def run_cross_validation(x, y, 
                         x_test_final, y_test_final,
                         model, 
                         CP_params,
                         n_splits=5, n_splits_CP=5, 
                         input_transform=True, output_transform=None, 
                         classification=False,
                         verbose=True, show_plots=True,save=False,which_data=' '):
    
    save_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries','bainite_boundaries', 'results','models',which_data)

    if save:
        variables = {
            "CP_params": CP_params,
            "n_splits": n_splits,
            "n_splits_CP": n_splits_CP,
            "input_transform": input_transform,
            "output_transform": output_transform,
            "classification": classification,
            "verbose": verbose,
        }
        with open(save_path/(model.__class__.__name__+"pipline_variables.json"), "w") as f:
            json.dump(variables,f)
    
    #stratified CV
    if isinstance(y, list):
        y_bin=np.array([np.mean(y[k]) for k in range(len(y))])   
    else:
        y_bin=y

    num_strat = 5  # Choose the number of bins
    kbins = KBinsDiscretizer(n_bins=num_strat, encode="ordinal", strategy="quantile")  
    y_binned = kbins.fit_transform(y_bin.reshape(-1, 1)).astype(int).flatten()  

    logger = setup_logging(verbose)
        
    if n_splits == 1:
        kf = StratifiedKFold(n_splits=n_splits+5, shuffle=True)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        
    if n_splits_CP == 1:
        kf_CP = StratifiedKFold(n_splits=n_splits_CP+1, shuffle=True)
    else:
        kf_CP = StratifiedKFold(n_splits=n_splits_CP, shuffle=True)
        
    results = []
    for i, (train_index_out, val_index_out) in enumerate(kf.split(x,y_binned)):


        logger.info(f'Validation fold (outer) {i+1}/{n_splits}')
        
        
        for j, (train_index_CP, val_index_CP) in enumerate(kf_CP.split(train_index_out,y_binned[train_index_out])):
            
            logger.info(f'    Validation fold (inner) {j+1}/{n_splits_CP}')

            
            if n_splits == 1:
                train_index_out = np.concatenate((train_index_out, val_index_out))
                val_index_out = []
                
            train_index = train_index_out[train_index_CP].astype(int)
            val_index = train_index_out[val_index_CP].astype(int)
            test_index = val_index_out
            
            # Create pipeline
            pipeline = ModelPipeline(model, model_name='YourModel', input_transform=input_transform, output_transform=output_transform)
            pipeline.task_type = 'classification' if classification else 'regression'
            
            # Prepare and transform data
            X_train, y_train, X_calib, y_calib, X_train_proper, y_train_proper, X_test, y_test = pipeline.prepare_data(x, y, train_index, val_index, test_index)
            if save:
                np.save(save_path/(model.__class__.__name__+str(j)+"_X_train_proper.json"),X_train_proper)
                np.save(save_path/(model.__class__.__name__+str(j)+"_X_calib.json"),X_calib)


            if x_test_final is not None:
                
                if X_test is not None:
                    X_calib = np.concatenate((X_calib, X_test), axis=0)
                    y_calib = np.concatenate((y_calib, y_test), axis=0)
                
                X_test = x_test_final
                y_test = y_test_final
            
            logger.info(f'        Train size: {len(X_train_proper)}, Calib size: {len(X_calib)}')
            if X_test is not None:
                logger.info(f'        Test size: {len(X_test)}')
                        
            # Fit model
            pipeline.fit(X_train_proper, y_train_proper)


            if save:
                1#pipeline.model.save_model(save_path/(model.__class__.__name__+str(j)))
            
            # Get distance function
            pipeline.calculate_distances = CalculateDistances(n_neighbors=CP_params['n_neighbors'],
                                                    distance_loc=CP_params['distance_loc'],
                                                    distance_metric=CP_params['distance_metric'])

            X_test_distance = X_calib if X_test is None else np.concatenate((X_calib, X_test), axis=0)

            pipeline.calculate_distances.fit(X_train_proper, X_test_distance)
            distance_train = pipeline.calculate_distances.distance_transform(X_train_proper)
            distance_calib = pipeline.calculate_distances.distance_transform(X_calib)
            distance_test = pipeline.calculate_distances.distance_transform(X_test)
            
            logger.info(f'        Distance train: {distance_train.mean():.2f}, Distance calib: {distance_calib.mean():.2f}')

            if X_test is not None:
                logger.info(f'        Distance test: {distance_test.mean():.2f}')
            
            # Compute quantile scores
            pipeline.quantile_scores(X_calib, y_calib, coverage=CP_params['coverage'], distance=distance_calib, distance_scale=CP_params['distance_scale'])
            



            if save:
                CV_dict = {
                    "distance_scale": pipeline.distance_scale,
                    "distance_mean": pipeline.distance_mean,
                    "CP_quantiles_distance": pipeline.CP_quantiles_distance
                }
                with open(save_path/(model.__class__.__name__+str(j)+"_dist.json"), "w") as f:
                    json.dump(CV_dict,f)
                    

        

            
            # Get interval predictions
            prediction_dict_train = pipeline.get_interval_predictions(X_train_proper, CP_params['coverage'], distance=distance_train)
            prediction_dict_calib = pipeline.get_interval_predictions(X_calib, CP_params['coverage'], distance=distance_calib)
            
            # Compute metrics           
            error_train = pipeline.compute_errors(y_train_proper, prediction_dict_train)
            error_calib = pipeline.compute_errors(y_calib, prediction_dict_calib)
            
            if show_plots:
                if pipeline.task_type == 'classification':
                    plot_confusion_matrix(y_train_proper, prediction_dict_train['y_pred'], title='Train set')
                    plot_confusion_matrix(y_calib, prediction_dict_calib['y_pred'], title='Calibration set')
                else:
                    plot_true_vs_pred(y_train_proper, prediction_dict_train['y_pred'], title='Train set', 
                                      log=output_transform=='log')
                    plot_true_vs_pred(y_calib, prediction_dict_calib['y_pred'], title='Calibration set', 
                                      log=output_transform=='log')
                    plot_predictions(y_calib, prediction_dict_calib['y_pred'], prediction_dict_calib['upper_CP'], prediction_dict_calib['lower_CP'], prediction_dict_calib['upper_CP_distance'], prediction_dict_calib['lower_CP_distance'], title='Calibration set', log=output_transform=='log')
            
            if X_test is not None:
                prediction_dict_test = pipeline.get_interval_predictions(X_test, CP_params['coverage'], distance=distance_test)
                
                error_test = pipeline.compute_errors(y_test, prediction_dict_test)

                if show_plots:
                    if pipeline.task_type == 'classification':
                        if y_test is not None:
                            plot_confusion_matrix(y_test, prediction_dict_test['y_pred'], title='Test set')
                            plot_calibration_curve(y_test, prediction_dict_test['y_pred'], title='Test set')
                    else:
                        plot_predictions(y_test, prediction_dict_test['y_pred'], prediction_dict_test['upper_CP'], prediction_dict_test['lower_CP'], prediction_dict_test['upper_CP_distance'], prediction_dict_test['lower_CP_distance'], title='Test set', ls='--', marker=None, log=output_transform=='log')#, xlim=[0, 500])
            else:
                prediction_dict_test = {}
                error_test = {}
            

            # Store results
            results.append({
                'CV_fold': i,
                'CP_CV_fold': j,
                'pipeline': copy.deepcopy(pipeline),
                # model name
                'distance_metric': CP_params['distance_metric'],
                'distance_loc': CP_params['distance_loc'],
                # indices
                'train_index': train_index,
                'val_index': val_index,
                'test_index': test_index,
                # targets
                'y_train': y_train_proper,
                'y_calib': y_calib,
                'y_test': y_test,
                # predictions
                **{key + '_train': value for key, value in prediction_dict_train.items()},
                **{key + '_calib': value for key, value in prediction_dict_calib.items()},
                **{key + '_test': value for key, value in prediction_dict_test.items()},
                # errors
                'error_train': error_train,
                'error_calib': error_calib,
                'error_test': error_test, 
            })



            
            if n_splits_CP == 1:
                if j == 0:
                    break
        
        if i == n_splits-1:
            break



    df = pd.DataFrame(results)


    return df




def run_prediction(x_test_final,path,model):
    
    with open(path+"pipline_variables.json", "r") as f:
        variables = json.load(f)

    x = variables["x"]
    y = variables["y"]
    CP_params = variables["CP_params"]
    n_splits_CP = variables["n_splits_CP"]
    input_transform = variables["input_transform"]
    output_transform = variables["output_transform"]
    classification = variables["classification"]

        
        
    results = []

        
    for j in range(n_splits_CP):



        
        #load model?
        model=model.load_model(path)

        #load this
        pipeline = ModelPipeline(model, model_name='YourModel', input_transform=input_transform, output_transform=output_transform)
        pipeline.task_type = 'classification' if classification else 'regression'
        

        #load this
        X_train_proper=0
        X_calib=0

        

        # load here          
        # Fit model
        


        # Get distance function
        pipeline.calculate_distances = CalculateDistances(n_neighbors=CP_params['n_neighbors'],
                                                distance_loc=CP_params['distance_loc'],
                                                distance_metric=CP_params['distance_metric'])
        



        X_test_distance = np.concatenate((X_calib, x_test_final), axis=0) 
        pipeline.calculate_distances.fit(X_train_proper, X_test_distance)

        distance_test = pipeline.calculate_distances.distance_transform(x_test_final)
        

        with open(save_path/(model.__class__.__name__+str(j)+"_dist.json"), "r") as f:
            data_dict = json.load(f)

        pipeline.distance_scale = variables["distance_scale"]
        pipeline.distance_scale = variables["distance_mean"]
        pipeline.distance_scale = variables["CP_quantiles_distance"]                
        
   
        prediction_dict_test = pipeline.get_interval_predictions(x_test_final, CP_params['coverage'], distance=distance_test)
        

        results.append({
        #'CP_CV_fold': j,
        #'y_train': y_train_proper,
        #'y_calib': y_calib,
        #'y_test': y_test,
        # predictions
        #**{key + '_test': value for key, value in prediction_dict_test.items()},
    })    

    return results
