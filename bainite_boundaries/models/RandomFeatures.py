from sklearn.linear_model import Ridge
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.kernel_approximation import RBFSampler

from sklearn.decomposition import PCA
import numpy as np

class RandomFeatureModel:
    def __init__(self, n_components=100, apply_pca=False, explained_variance=0.999, transform_features_flag=False, random_state=1):
        print("RandomFeatureModel")
        
        self.n_components = n_components
        self.apply_pca = apply_pca
        self.explained_variance = explained_variance
        self.random_state = random_state
        self.model = Ridge()
        self.random_features = GaussianRandomProjection(n_components=n_components)
        self.random_features = RBFSampler(n_components=n_components)
        self.pca = None
        self.transform_features_flag = transform_features_flag
    
    def transform_features(self, x):
        feature = x[:, -1:]
        x_transformed = x
        for i in [2,4,8]:
            x_transformed = np.hstack([x_transformed, np.log(feature)** i])
            
        x_transformed = np.hstack([x_transformed, np.log(feature)])
        return x_transformed
    
    def fit(self, train_x, train_y):
        
        # Apply PCA if required
        if self.apply_pca:
            self.pca = PCA(self.explained_variance, random_state=self.random_state)
            train_x = self.pca.fit_transform(train_x)
        
        # Transform the data to random features
        if self.transform_features_flag:
            train_x = self.transform_features(train_x)
            
        train_x_random = self.random_features.fit_transform(train_x)
        
        # Fit the Ridge regression model
        self.model.fit(train_x_random, train_y)
        
    def predict(self, test_x):
        
        if self.transform_features_flag:
            test_x = self.transform_features(test_x)
        
        # Transform the test data to random features
        test_x_random = self.random_features.transform(test_x)
        
        # Apply PCA if it was used in training
        if self.apply_pca and self.pca is not None:
            test_x_random = self.pca.transform(test_x_random)
        
        return self.model.predict(test_x_random)
    