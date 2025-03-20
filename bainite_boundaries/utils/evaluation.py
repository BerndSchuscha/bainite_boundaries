import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error
    
def evaluate_df_results( results_df_agg, kCP=True ):
    
    predicted_values = np.concatenate([row['prediction'].mean() for index, row in results_df_agg])
    ground_truth_values = np.concatenate([row['ground_truth'].mean() for index, row in results_df_agg])
    upper_test_CP = np.concatenate([row['upper_test_CP'].mean() for index, row in results_df_agg])
    lower_test_CP = np.concatenate([row['lower_test_CP'].mean() for index, row in results_df_agg])
    upper_test_CP_weighted = np.concatenate([row['upper_test_CP_weighted'].mean() for index, row in results_df_agg])
    lower_test_CP_weighted = np.concatenate([row['lower_test_CP_weighted'].mean() for index, row in results_df_agg])

    # compute global results
    coverage_CP = (ground_truth_values < upper_test_CP) & (ground_truth_values > lower_test_CP)
    coverage_CP_weighted = (ground_truth_values < upper_test_CP_weighted) & (ground_truth_values > lower_test_CP_weighted)
    MAPE = np.abs(ground_truth_values - predicted_values) / ground_truth_values
    MALE = np.abs(np.log10(ground_truth_values) - np.log10(predicted_values))
    interval_width = (np.log10(upper_test_CP) - np.log10(lower_test_CP))
    interval_width_weighted = (np.log10(upper_test_CP_weighted) - np.log10(lower_test_CP_weighted))
    r2 = r2_score(ground_truth_values, predicted_values)
    mae = mean_absolute_error(ground_truth_values, predicted_values)

    # compute mean results for each fold
    total_coverage_CP = results_df_agg['coverage_CP'].mean()
    total_coverage_CP_weighted = results_df_agg['coverage_CP_weighted'].mean()

    total_MAPE = results_df_agg['mape'].mean()
    total_MALE = results_df_agg['male'].mean()
    total_interval_width = results_df_agg['interval_width'].mean()
    total_interval_width_weighted = results_df_agg['interval_width_weighted'].mean()

    print("*"*50 + "Global Results " + "*"*50)
    print(f"MALE: {MALE.mean():.2f}")
    print(f"Coverage: {coverage_CP.mean():.2f}")
    print(f"Interval width: {interval_width.mean():.2f}")
    print(f"Coverage weighted: {coverage_CP_weighted.mean():.2f}")
    print(f"Interval width weighted: {interval_width_weighted.mean():.2f}")
    print(f"MAPE: {MAPE.mean():.2f}")
    print(f"R2: {r2:.2f}")
    print(f"MAE: {mae:.2f}")

    print("*"*50 + "Averaged Results " + "*"*50)
    print(f"total_MALE: {total_MALE.mean():.2f} +- {total_MALE.std():.2f}")
    print(f"Total coverage: {total_coverage_CP.mean():.2f}")
    print(f"total_interval_width: {total_interval_width.mean():.2f} +- {total_interval_width.std():.2f}")
    print(f"Total coverage weighted: {total_coverage_CP_weighted.mean():.2f} +- {total_coverage_CP_weighted.std():.2f}")
    print(f"total_interval_width_weighted: {total_interval_width_weighted.mean():.2f} +- {total_interval_width_weighted.std():.2f}")
    print(f"total_MAPE: {total_MAPE.mean():.2f} +- {total_MAPE.std():.2f}")


def evaluate_final_predictions( results_df):
    print(f"Calibration MALE = {results_df['male'].mean():.2f} +- {results_df['male'].std():.2f}")
    y_pred = results_df['prediction'].mean()
    constraint_percentage = np.mean( y_pred > 10_000 ) * 100
    print(f"Constraint percentage: {constraint_percentage:.2f}%")
    y_pred = results_df['lower_test_CP'].mean()
    constraint_percentage = np.mean( y_pred > 10_000 ) * 100
    print(f"Constraint percentage lower quantile: {constraint_percentage:.2f}%")
    y_pred = results_df['lower_test_CP_weighted'].mean()
    constraint_percentage = np.mean( y_pred > 10_000 ) * 100
    print(f"Constraint percentage lower quantile weighted: {constraint_percentage:.2f}%")
    y_pred = results_df['upper_test_CP'].mean()
    constraint_percentage = np.mean( y_pred > 10_000 ) * 100
    print(f"Constraint percentage upper quantile: {constraint_percentage:.2f}%")
    y_pred = results_df['upper_test_CP_weighted'].mean()
    constraint_percentage = np.mean( y_pred > 10_000 ) * 100
    print(f"Constraint percentage upper quantile weighted: {constraint_percentage:.2f}%")

def plot_train_calib_vs_actual( results_df, which_data='Bainite', model_name_1='Polynomial Regression', distance=False):
    plt.figure(figsize=(6, 6))
    
    for index, row in results_df.iterrows():
        predicted_train_values = row['prediction_train']
        ground_truth_train_values = row['ground_truth_train']
        predicted_calib_values = row['prediction_calib']
        ground_truth_calib_values = row['ground_truth_calib']
    
        
        plt.scatter(ground_truth_train_values, predicted_train_values, label='Train', alpha=0.5, color='C0')
        plt.scatter(ground_truth_calib_values, predicted_calib_values, label='Calib', alpha=0.5, color='C1')
        
        plt.plot([0, 10000], [0, 10000], color='k', linestyle='--', alpha=0.6)

        plt.xlabel('Actual transformation time', fontsize=20)
        plt.ylabel('Predicted transformation time', fontsize=20)
        # log log scale
        if which_data in ['Bainite', 'Ferrite']:
            plt.xscale('log')
            plt.yscale('log')
        plt.axis('equal')
        plt.title(f"{which_data}, {model_name_1}", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        plt.tight_layout()
    plt.show()
    
    
def plot_pred_vs_actual( results_df_agg, which_data='Bainite', model_name_1='Polynomial Regression', distance=False):

    predicted_values = np.concatenate([row['prediction'].mean() for index, row in results_df_agg])
    ground_truth_values = np.concatenate([row['ground_truth'].mean() for index, row in results_df_agg])
    upper_test_CP = np.concatenate([row['upper_test_CP'].mean() for index, row in results_df_agg])
    lower_test_CP = np.concatenate([row['lower_test_CP'].mean() for index, row in results_df_agg])
    upper_test_CP_weighted = np.concatenate([row['upper_test_CP_weighted'].mean() for index, row in results_df_agg])
    lower_test_CP_weighted = np.concatenate([row['lower_test_CP_weighted'].mean() for index, row in results_df_agg])
    
    # compute global results
    coverage_CP = (ground_truth_values < upper_test_CP) & (ground_truth_values > lower_test_CP)
    coverage_CP_weighted = (ground_truth_values < upper_test_CP_weighted) & (ground_truth_values > lower_test_CP_weighted)
    
    plt.figure(figsize=(6, 6))
    
    if distance:
        eb1=plt.errorbar(ground_truth_values[ coverage_CP_weighted], predicted_values[ coverage_CP_weighted], yerr=[predicted_values[ coverage_CP_weighted] - lower_test_CP_weighted[ coverage_CP_weighted], upper_test_CP_weighted[ coverage_CP_weighted] - predicted_values[ coverage_CP_weighted]], fmt='x', color='green', alpha=0.25)
        eb2=plt.errorbar(ground_truth_values[~coverage_CP_weighted], predicted_values[~coverage_CP_weighted], yerr=[predicted_values[~coverage_CP_weighted] - lower_test_CP_weighted[~coverage_CP_weighted], upper_test_CP_weighted[~coverage_CP_weighted] - predicted_values[~coverage_CP_weighted]], fmt='x', color='red', alpha=0.25)
    else:
        plt.errorbar(ground_truth_values[ coverage_CP], predicted_values[ coverage_CP], yerr=[predicted_values[ coverage_CP] - lower_test_CP[ coverage_CP], upper_test_CP[ coverage_CP] - predicted_values[ coverage_CP]], fmt='x', color='green', alpha=0.25)
        plt.errorbar(ground_truth_values[~coverage_CP], predicted_values[~coverage_CP], yerr=[predicted_values[~coverage_CP] - lower_test_CP[~coverage_CP], upper_test_CP[~coverage_CP] - predicted_values[~coverage_CP]], fmt='x', color='red', alpha=0.25)
        
    max_value = max(max(ground_truth_values), max(predicted_values))
    min_value = min(min(ground_truth_values), min(predicted_values))
    
    plt.plot([0, max_value], [0, max_value], color='k', linestyle='--', alpha=0.6)
    plt.xlabel('Actual transformation time', fontsize=20)
    plt.ylabel('Predicted transformation time', fontsize=20)
    # log log scale
    if which_data in ['Bainite', 'Ferrite']:
        plt.xscale('log')
        plt.yscale('log')
    plt.axis('equal')
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.title(f"{which_data}, {model_name_1}, coverage: {coverage_CP_weighted.mean():.2f}%", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()
        
        

def plot_predictions(results_df, which_data='Bainite', model_name_1='Polynomial Regression'):
    
    y_pred = results_df['prediction'].mean()
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.plot(results_df['prediction'].mean(), label='Prediction')
    ax.fill_between(np.arange(len(y_pred)), 
                    results_df['prediction'].mean() - np.array(results_df['prediction']).std(), results_df['prediction'].mean() + np.array(results_df['prediction']).std(),
                    alpha=0.5, label='Prediction std')
    ax.fill_between(np.arange(len(y_pred)),
                    results_df['lower_test_CP'].mean(), results_df['upper_test_CP'].mean(),
                    alpha=0.5, label='Prediction interval')
    ax.fill_between(np.arange(len(y_pred)),
                    results_df['lower_test_CP_weighted'].mean(), results_df['upper_test_CP_weighted'].mean(), alpha=0.5, label='Prediction interval weighted')
    if which_data in ['Bainite', 'Ferrite']:
        ax.set_yscale('log')
    ax.legend( bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.set_ylim(0.1, 10000)
    # ax.set_xlim(0, 500)
    plt.show()
    
   
def plot_kfold_results(results_df_agg, which_data='Bainite', model_name_1='Polynomial Regression'):
    
    cols = 3
    N_plot = len(results_df_agg['Material (test)'].mean())
    rows = np.ceil(N_plot / cols ).astype(int)

    # Create a large figure to hold all subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing


    # Plotting after training
    for index, row in results_df_agg:
        
        n_model = row['Material (test)'].mean()
        ax = axes[index]
        
        y_test = row['ground_truth'].mean()
        y_pred = row['prediction'].mean()
        lower_test_CP = row['lower_test_CP'].mean()
        upper_test_CP = row['upper_test_CP'].mean()
        lower_test_CP_weighted = row['lower_test_CP_weighted'].mean()
        upper_test_CP_weighted = row['upper_test_CP_weighted'].mean()
        # temperature = row['temperature'].mean()
        temperature = np.arange(len(y_test))
        
        ax.fill_between(temperature, lower_test_CP, upper_test_CP, alpha=0.5, color='C0', label=model_name_1, interpolate=True)
        ax.fill_between(temperature, lower_test_CP_weighted, upper_test_CP_weighted, alpha=0.5, color='C1', label=f'{model_name_1} weighted', interpolate=True)
        ax.scatter(temperature, y_test, marker='x', c='r', label='Target $y$')  
        ax.scatter(temperature, y_pred, label=f'Prediction $\hat{{y}}=\mu^{model_name_1}$', c='C0')
        # ax.set_xlabel('$T~[°C]$')
        # remove x-ticks
        ax.set_xticks([])
        ax.set_ylabel('$t_{90\%}$')
        ax.legend(ncol=1)
        
        # log scale
        ax.set_yscale('log')
        # ax.set_ylim(1, 5000)
        
        ax.set_title(f'Test material: {row["Material (test)"].mean()},'
                    f'\n MAPLE: {row["mape"].mean():.2f}, \n MALE {row["male"].mean():.2f}, \n Coverage: {row["coverage_CP"].mean():.2f}')

    # Ensure any unused subplots are not visible if N is not a perfect square
    for i in range(N_plot, len(axes)):
        fig.delaxes(axes[i])

    # fig.suptitle(f"{model_name_1}", fontsize=16, y=1.02)
    fig.tight_layout()
    plt.show()