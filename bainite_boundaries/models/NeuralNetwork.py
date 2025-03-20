import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class NeuralNetworkModel:
    def __init__(self, 
                 hidden_size, 
                 apply_pca=False, 
                 explained_variance=0.999, 
                 transformations=None, 
                 random_state=1):
        self.apply_pca = apply_pca
        self.explained_variance = explained_variance
        self.random_state = random_state
        self.pca = None
        
        if transformations is None:
            self.transform_features_flag = False
        else:
            # Get the list of descriptions
            transformation_descriptions = list(transformations.keys())
            print(transformation_descriptions)
            self.transformations = list(transformations.values())
            self.transform_features_flag = True
        
        # Define the MLPRegressor
        self.model = MLPRegressor(hidden_layer_sizes=hidden_size, 
                                  max_iter=1500, 
                                  random_state=self.random_state, 
                                  activation='relu', 
                                  solver='adam', 
                                  learning_rate='adaptive', 
                                  alpha=0.0001, 
                                  early_stopping=True, 
                                  validation_fraction=0.05, 
                                  n_iter_no_change=100)

        
    def transform_features(self, x):
        feature = x[:, -1:]  # Extract the last feature (T)
        x_transformed = x
        
        # Apply each transformation specified in the transformations list
        for transformation in self.transformations:
            transformed_feature = transformation(feature)            
            x_transformed = np.hstack([x_transformed, transformed_feature])
        
        return x_transformed
    
    def fit(self, train_x, train_y):
        if self.apply_pca:
            self.pca = PCA(self.explained_variance, random_state=self.random_state)
            train_x = self.pca.fit_transform(train_x)
            
        if self.transform_features_flag: 
            train_x = self.transform_features(train_x)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        if self.apply_pca and self.pca is not None:
            test_x = self.pca.transform(test_x)
        if self.transform_features_flag: 
            test_x = self.transform_features(test_x)
        predictions = self.model.predict(test_x)
        return predictions


class NeuralNetworkClassification:
    def __init__(self, hidden_size=(500), max_iter=5, random_state=1):
        self.model = None
        self.calibrated_model = None
        self.hidden_size = hidden_size
        self.max_iter = max_iter
        
    def fit(self, train_x, train_y):
        # Initialize MLPClassifier with given hyperparameters
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_size, max_iter=self.max_iter, random_state=1)
        
        # Split the data into training and calibration sets
        train_x, calibrate_x, train_y, calibrate_y = train_test_split(
            train_x, train_y, test_size=0.3, random_state=1, stratify=train_y
        )
        
        # Fit the base model
        self.model.fit(train_x, train_y)
        
        # Calibrate using Isotonic Regression
        self.calibrated_model = CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')
        self.calibrated_model.fit(calibrate_x, calibrate_y)

    def predict(self, test_x):
        return self.model.predict(test_x)
    
    def predict_proba(self, test_x):
        # Get calibrated probabilities using isotonic regression
        return self.calibrated_model.predict_proba(test_x)
