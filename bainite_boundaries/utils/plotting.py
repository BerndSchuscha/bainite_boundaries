import umap
#import umap.plot

from scipy import linalg
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve


color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(X, Y_, means, covariances, index, title):
    fig, splot = plt.subplots(figsize=(3.5, 2.5))
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.xticks(())
    plt.yticks(())
    
    
#def plot_umap(reducer, train_embedding, test_embedding, y_train_proper_log, X_test, n_model):
#    fig_umap, ax_umap = plt.subplots(figsize=(8, 6))
#    umap.plot.points(reducer, labels=y_train_proper_log, ax=ax_umap)
#    ax_umap.legend(handles=[], labels=[])
#    ax_umap.scatter(test_embedding[:, 0], test_embedding[:, 1], c='r', marker='x')
#    for i, txt in enumerate(X_test[:, -2]):
#        ax_umap.annotate(f'{txt:.0f}', (test_embedding[i, 0], test_embedding[i, 1]))
#    ax_umap.set_title(f'Material {n_model}')
#    plt.show()


def plot_predictions(y_true, y_pred, 
                    upper=None, lower=None, upper_distance=None, lower_distance=None, 
                    log=False,
                    title='Predictions', 
                    marker='x', ls='-', 
                    xlim=None, ylim=None):
    
    x_values = np.arange(len(y_pred))

    plt.figure(figsize=(3, 3))
    if y_true is not None:
        plt.plot(x_values, y_true, label='True', marker=marker, ls=ls)
    plt.plot(x_values, y_pred, label='Predicted', marker=marker, ls=ls)
    if upper is not None:
        plt.fill_between(x_values, lower, upper, alpha=0.2, color='blue', label='Confidence interval')
    if upper_distance is not None:
        plt.fill_between(x_values, lower_distance, upper_distance, alpha=0.2, color='red', label='Confidence interval (scaled)')
    if log:
        plt.yscale('log')
    plt.title(title)
    plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left' )
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()

def plot_true_vs_pred(y_true, y_pred, log=False, title='True vs Predicted'):
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.figure(figsize=(2, 2))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.axis('equal')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.title(title)
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting the confusion matrix
    plt.figure(figsize=(2, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()
    
def plot_calibration_curve(y_true, y_pred, title='Calibration Curve'):
    # Compute the calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
    
    # Plotting the calibration curve
    plt.figure(figsize=(2, 2))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(title)
    plt.legend()
    plt.show()
    

   
def plot_kfold_results(results_df, which_data='Bainite', model_name_1='Polynomial Regression', log=False):
    
    cols = 3
    N_plot = len(results_df['CV_fold'].unique())
    rows = np.ceil(N_plot / cols ).astype(int)

    # Create a large figure to hold all subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    fig1, axes1 = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes1 = axes1.flatten()  # Flatten to 1D array for easy indexing
    # Plotting after training
    for index in range(N_plot):
        
        row = results_df[results_df['CV_fold'] == index]
                
        n_model = row['CV_fold'].mean()
        ax = axes[index]
        ax1 = axes1[index]
        
        y_test = row['y_test'].mean()
        y_pred = np.array( row['y_pred_test'] )
        y_pred_mean = y_pred.mean(axis=0)
        y_pred_std = y_pred.std(axis=0)
        
        lower_CP_test = np.array( row['lower_CP_test'] ).mean(axis=0)
        upper_CP_test = np.array( row['upper_CP_test'] ).mean(axis=0)
        lower_CP_distance_test = np.array( row['lower_CP_distance_test'] ).mean(axis=0)
        upper_CP_distance_test = np.array( row['upper_CP_distance_test'] ).mean(axis=0)
        
        
        temperature = np.arange(len(y_pred_mean))
        if isinstance(y_test, np.ndarray):
            ax.scatter(temperature, y_test, marker='x', c='r', label='Target $y$')  
        ax.scatter(temperature, y_pred_mean, label=f'Prediction $\hat{{y}}=\mu^{{{model_name_1}}}$', c='C0')
        ax.fill_between(temperature, y_pred_mean - y_pred_std, y_pred_mean + y_pred_std, alpha=0.2, color='blue', label='CV std')
        ax.fill_between(temperature, lower_CP_test, upper_CP_test, alpha=0.2, color='green', label='CP')
        ax.fill_between(temperature, lower_CP_distance_test, upper_CP_distance_test, alpha=0.2, color='red', label='CP distance')
        # ax.set_xlabel('$T~[°C]$')
        # remove x-ticks
        ax.set_xticks([])
        ax.set_ylabel('$t_{90\%}$')            
        


        ax1.hist((y_test-y_pred_mean)/(lower_CP_distance_test-upper_CP_distance_test)*1.28,alpha=0.5,bins=50)
        ax1.set_title('standard devition = '+str(np.std((y_test-y_pred_mean)/(lower_CP_distance_test-upper_CP_distance_test)*1.28)))
        #ax.set_xticks([])
        #ax.set_ylabel('$t_{90\%}$')    


        MALE = np.array( [ row['MALE'] for row in row['error_test'] ])
        test_index = np.array(row['test_index'].to_list()) 
        if MALE[0] is not None:
            ax.set_title(f"MALE = {MALE.mean():.2f} +- {MALE.std():.2f}")
        else: 
            ax.set_title(f"")
            
        # log scale
        if log:
            ax.set_yscale('log')
        # ax.set_ylim(1, 5000)
        

    # Ensure any unused subplots are not visible if N is not a perfect square
    for i in range(N_plot, len(axes)):
        fig.delaxes(axes[i])

    # fig.suptitle(f"{model_name_1}", fontsize=16, y=1.02)
    fig.tight_layout()
    axes[0].legend(ncol = 2, loc='upper left', bbox_to_anchor=(0.5, 1.5))
    plt.show()

