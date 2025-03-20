import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn import mixture
import umap
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


from sklearn.mixture import GaussianMixture
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC


from scipy.spatial.distance import mahalanobis, cdist
from scipy.stats import wasserstein_distance, entropy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon, cdist

class CalculateDistances:
    def __init__(self, n_neighbors=5, distance_loc='PCA', distance_metric='knn'):
        self.n_neighbors = n_neighbors
        self.distance_loc = distance_loc
        self.distance_metric = distance_metric

    def transform(self, x):
        if self.distance_loc == 'PCA':
            x_transform = self.pca.transform(x)
        elif self.distance_loc == 'PCA_scaled':
            x_transform = self.scaler.transform(x)
            x_transform = self.pca.transform(x_transform)
        elif self.distance_loc == 'input':
            x_transform = x
        elif self.distance_loc == 'input_scaled':
            x_transform = self.scaler.transform(x)
        return x_transform

    def distance_fit(self, x, x_test=None):
        
        if self.distance_metric == 'knn':
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.knn.fit(x)

        elif self.distance_metric == 'likelihood_ratio':
            # Logistic regression for likelihood ratio
            n_test = x_test.shape[0] if x_test is not None else 0
            y = np.zeros(x.shape[0])
            X = x
            if x_test is not None:
                y = np.append(y, np.ones(n_test))
                X = np.vstack((X, x_test))
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            clf.fit(X, y)
            self.clf = clf

        elif self.distance_metric == 'mahalanobis':
            # Fit covariance matrix for Mahalanobis distance
            cov_estimator = EmpiricalCovariance()
            cov_estimator.fit(x)
            self.cov_inv = np.linalg.inv(cov_estimator.covariance_)
            self.mean = cov_estimator.location_

        elif self.distance_metric == 'mixture_model':
            # Gaussian mixture model
            optimal_components, gmm_model = select_optimal_components(x, max_components=10)
            self.model = gmm_model

        elif self.distance_metric == 'wasserstein':
            # Wasserstein requires a comparison dataset
            self.reference_distribution = np.mean(x, axis=0)
            
        elif self.distance_metric == 'wasserstein_truncated':
            self.calibration_embeddings = x
            self.min_distance = None
            self.max_distance = None
            self._calculate_truncation_bounds(x, truncation_quantile=0.01)

        elif self.distance_metric == 'lof':
            self.lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
            self.lof.fit(x)

        elif self.distance_metric == 'cosine':
            # Cosine similarity requires embeddings of the reference dataset
            self.reference_embeddings = x

        elif self.distance_metric == 'jensen_shannon':
            # Jensen-Shannon requires a reference distribution (mean of x)
            self.reference_distribution = np.mean(x, axis=0)

        elif self.distance_metric == 'energy':
            # Energy distance requires a reference distribution (entire dataset)
            self.reference_distribution = x

        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _calculate_truncation_bounds(self, x, truncation_quantile=0.01):
        # Determine truncation bounds based on the quantile
        lower_quantile = np.percentile(x, truncation_quantile * 100, axis=0)
        upper_quantile = np.percentile(x, (1 - truncation_quantile) * 100, axis=0)

        self.truncated_x = np.clip(x, lower_quantile, upper_quantile)
        self.min_distance = 0
        self.max_distance = np.linalg.norm(upper_quantile - lower_quantile)


    def fit(self, x, x_test=None):
        if self.distance_loc == 'PCA':
            self.pca = PCA(0.99)
            self.pca.fit(x)
        elif self.distance_loc == 'PCA_scaled':
            self.pca = PCA(0.99)
            self.scaler = StandardScaler()
            self.scaler.fit(x)
            x_transform = self.scaler.transform(x)
            self.pca.fit(x_transform)
        elif self.distance_loc == 'input_scaled':
            self.scaler = StandardScaler()
            self.scaler.fit(x)
        else:
            self.pca = None
            self.scaler = None

        x_transform = self.transform(x)

        if x_test is not None:
            x_test_transform = self.transform(x_test)
        else:
            x_test_transform = None

        self.distance_fit(x_transform, x_test_transform)

    def distance_transform(self, x):
        if x is None:
            return None

        x_transform = self.transform(x)

        if self.distance_metric == 'knn':
            distances = self.knn.kneighbors(x_transform)[0]
            distances = np.mean(distances, axis=1)

        elif self.distance_metric == 'likelihood_ratio':
            likelihood = self.clf.predict_proba(x_transform).clip(0.01, 0.99)
            distances = likelihood[:, 1] / likelihood[:, 0]

        elif self.distance_metric == 'mahalanobis':
            distances = [mahalanobis(xi, self.mean, self.cov_inv) for xi in x_transform]
            distances = np.array(distances)

        elif self.distance_metric == 'mixture_model':
            distances = -self.model.score_samples(x_transform)

        elif self.distance_metric == 'wasserstein':
            distances = [wasserstein_distance(self.reference_distribution, xi) for xi in x_transform]
            distances = np.array(distances)
            
        elif self.distance_metric == 'wasserstein_truncated':
            
            x_transform = self.transform(x)
            truncated_x_transform = np.clip(x_transform, self.truncated_x.min(axis=0), self.truncated_x.max(axis=0))

            distances = np.array([
                wasserstein_distance(self.calibration_embeddings.flatten(), sample.flatten())
                for sample in truncated_x_transform
            ])

        elif self.distance_metric == 'lof':
            distances = self.lof.score_samples(x_transform)
            
        elif self.distance_metric == 'cosine':
            distances = 1 - cosine_similarity(x_transform, self.reference_embeddings)
            distances = distances.mean(axis=1)

        elif self.distance_metric == 'jensen_shannon':
            distances = np.array([jensenshannon(self.reference_distribution, xi) for xi in x_transform])

        elif self.distance_metric == 'energy':
            distances = cdist(x_transform, self.reference_distribution, metric='euclidean').mean(axis=1)

        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        return distances


    


def apply_pca(X_train_proper, X_test, X_calib, explained_variance=0.999):
    pca = PCA(explained_variance)
    pca.fit(X_train_proper)
    X_train_proper_pca = pca.transform(X_train_proper)
    X_test_pca = pca.transform(X_test)
    X_calib_pca = pca.transform(X_calib)
    return pca, X_train_proper_pca, X_test_pca, X_calib_pca

def apply_umap(X_train_proper, X_test, X_calib, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    train_embedding = reducer.fit_transform(X_train_proper)
    test_embedding = reducer.transform(X_test)
    calib_embedding = reducer.transform(X_calib)
    return reducer, train_embedding, test_embedding, calib_embedding

def apply_lof(train_embedding, test_embedding, calib_embedding, n_neighbors=5):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(train_embedding)
    negative_lof_train = lof.score_samples(train_embedding)
    negative_lof_test = lof.score_samples(test_embedding)
    negative_lof_calib = lof.score_samples(calib_embedding)
    return negative_lof_train, negative_lof_test, negative_lof_calib

def plot_PCA(pca, X_train_proper_pca, X_test_pca, X_calib_pca, y_train_proper=None, y_calib=None, y_test=None):
    
    if pca.n_components_ == 2:
    
        if y_train_proper is not None:
        
            fig, ax = plt.subplots(figsize=(3,3))
            plt.scatter(X_train_proper_pca[:,0], X_train_proper_pca[:,1], c=y_train_proper, cmap='viridis', s=10)
            plt.scatter(X_calib_pca[:,0], X_calib_pca[:,1], c=y_calib, cmap='viridis', s=50, marker='+')
            plt.scatter(X_test_pca[:,0], X_test_pca[:,1], c=y_test, cmap='viridis', s=100, marker='x')
            plt.colorbar()
            plt.title('Train')
            plt.show()
            
    if pca.n_components_ == 1:
        
        if y_train_proper is not None:
        
            fig, ax = plt.subplots(figsize=(3,3))
            plt.scatter(X_train_proper_pca, np.zeros_like(X_train_proper_pca), c=y_train_proper, cmap='viridis', s=10)
            plt.scatter(X_calib_pca, np.zeros_like(X_calib_pca), c=y_calib, cmap='viridis', s=50, marker='+')
            plt.scatter(X_test_pca, np.zeros_like(X_test_pca), c=y_test, cmap='viridis', s=100, marker='x')
            plt.colorbar()
            plt.title('Train')
            plt.show()  



def select_optimal_components(X, max_components=10, random_state=42):
    """
    Select the optimal number of components for GMM using BIC.

    Args:
        X (numpy.ndarray): The data to fit.
        max_components (int): The maximum number of components to test.
        random_state (int): Random state for reproducibility.

    Returns:
        int: Optimal number of components.
        GaussianMixture: Fitted GMM with the optimal number of components.
    """
    bic_scores = []
    aic_scores = []
    models = []

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))
        models.append(gmm)

    # Select the optimal number of components based on BIC
    optimal_components = np.argmin(bic_scores) + 1
    optimal_model = models[np.argmin(bic_scores)]

    return optimal_components, optimal_model


def apply_gmm(train_embedding, test_embedding, calib_embedding, n_components_list=[5, 10, 20, 50, 100]):
    X = train_embedding
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0, reg_covar=0.1) for n in n_components_list]
    aics = [model.fit(X).aic(X) for model in models]
    best_model = models[np.argmin(aics)]
    
    gmm_score_train = best_model.score_samples(train_embedding)
    gmm_score_test = best_model.score_samples(test_embedding)
    gmm_score_calib = best_model.score_samples(calib_embedding)
    
    gmm_score_train -= gmm_score_train.mean()
    gmm_score_test -= gmm_score_train.mean()
    gmm_score_calib -= gmm_score_train.mean()
    
    return gmm_score_train, gmm_score_test, gmm_score_calib
