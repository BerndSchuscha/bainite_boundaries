import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.decomposition import PCA
from utils.data_processing import get_lambda_source
from scipy.optimize import minimize
import joblib

from sklearn.model_selection import GridSearchCV

class LinearRegressionModel:
    def __init__(self, apply_pca=False, explained_variance=0.999, 
                 transformations=None, random_state=1, alpha=1, search_alphas=None):
        self.apply_pca = apply_pca
        self.explained_variance = explained_variance
        self.random_state = random_state
        self.alpha = alpha
        self.search_alphas = search_alphas  # List of alphas for hyperparameter search
        self.model = Ridge(alpha=alpha)
        self.pca = None
        
        if transformations is None:
            self.transform_features_flag = False
        else:
            # Get the list of descriptions
            transformation_descriptions = list(transformations.keys())
            print(transformation_descriptions)
            self.transformations = list(transformations.values())
            self.transform_features_flag = True
            
    def transform_features(self, x):
        feature = x[:, -1:]  # Extract the last feature (T)
        x_transformed = x
        
        # Apply each transformation specified in the transformations list
        for transformation in self.transformations:
            transformed_feature = transformation(feature)            
            x_transformed = np.hstack([x_transformed, transformed_feature])
        
        return x_transformed
    
    def fit(self, train_x, train_y, cv=5):
        if self.apply_pca:
            self.pca = PCA(self.explained_variance, random_state=self.random_state)
            train_x = self.pca.fit_transform(train_x)
            
        if self.transform_features_flag: 
            train_x = self.transform_features(train_x)
            
        if self.search_alphas is not None:
            # Perform Grid Search to find the best alpha
            param_grid = {'alpha': self.search_alphas}
            grid_search = GridSearchCV(Ridge(), param_grid, cv=cv, scoring='neg_mean_squared_error')
            grid_search.fit(train_x, train_y)
            
            # Update the model with the best alpha
            self.alpha = grid_search.best_params_['alpha']
            print(f"Optimal alpha: {self.alpha}")
            self.model = Ridge(alpha=self.alpha)
        
        # Fit the final model
        self.model.fit(train_x, train_y)
        
    def predict(self, test_x):
        if self.apply_pca and self.pca is not None:
            test_x = self.pca.transform(test_x)
        if self.transform_features_flag: 
            test_x = self.transform_features(test_x)
        y=self.model.predict(test_x)
        return y
    
    def save_model(self,path):
        joblib.dump(self.model,path/"ridge_model.pkl")
        joblib.dump(self.pca,path/ "pca_model.pkl")
        keys_list = list(self.transformations.keys())
        joblib.dump(keys_list, "keys.pkl")


class LinearRegressionModel_mixeddata(LinearRegressionModel):
    def __init__(self, apply_pca=False, explained_variance=0.999, 
                 transformations=None, random_state=1,ycata=None,alpha=1,datatype=None, search_alphas=None):
        
        super().__init__( apply_pca, explained_variance, 
                 transformations, random_state)
        self.ycata=ycata
        self.alpha=alpha    
        self.datatype=datatype

    def _loss_function(self, coef, X, y):
        y_pred = X @ coef

        diff=y - y_pred

        L=np.logical_and(self.ycata==True,diff<0)

        loss = ((diff) ** 2)

        loss[L]=0

        loss=loss.mean()
        # Add L2 regularization term
        loss += self.alpha * np.sum(coef ** 2)
        return loss

    def fit(self, X, y):
        if self.datatype==None:
            super().fit(X,y)
            return self

        else:

            n_features = X.shape[1]
            initial_coef = np.zeros(n_features)

            # Minimize custom loss with L2 regularization
            result = minimize(self._loss_function, initial_coef, args=(X, y))
            self.coef_ = result.x
            return self
    

class LinearClassificationModel:
    def __init__(self, apply_pca=False, explained_variance=0.999, random_state=1):
        self.apply_pca = apply_pca
        self.explained_variance = explained_variance
        self.random_state = random_state
        self.model = RidgeClassifier()  # You can switch to LogisticRegression if needed
        self.pca = None

    def fit(self, train_x, train_y):
        if self.apply_pca:
            self.pca = PCA(self.explained_variance, random_state=self.random_state)
            train_x = self.pca.fit_transform(train_x)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        if self.apply_pca and self.pca is not None:
            test_x = self.pca.transform(test_x)
        return self.model.predict(test_x)
    
    def predict_proba(self, test_x):
        if self.apply_pca and self.pca is not None:
            test_x = self.pca.transform(test_x)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(test_x)
        else:
            scores = self.model.decision_function(test_x)
            probs = np.exp(scores) / (1 + np.exp(scores))
            return np.array([1 - probs, probs]).T

    def quantile_scores(self, calib_x, calib_y, coverage=0.8, distance=None):
        # Get the predicted probabilities
        proba_pred = self.predict_proba(calib_x)
        # Nonconformity scores for classification are based on 1 - max probability
        nonconformity_scores = 1 - np.max(proba_pred, axis=1)
        
        # Apply distance normalization if a distance measure is provided
        if distance is not None:
            nonconformity_scores = nonconformity_scores / distance
        
        # Compute the quantile score based on the nonconformity scores
        n = len(nonconformity_scores)
        quantile_scores = np.quantile(nonconformity_scores, coverage * (n + 1) / n)
        self.CP_quantiles = quantile_scores

    def predict_interval(self, test_x, coverage=0.9, distance=None):
        # Get predicted probabilities for the test set
        proba_pred = self.predict_proba(test_x)
        nonconformity_scores = 1 - np.max(proba_pred, axis=1)
        
        # Apply distance normalization if a distance measure is provided
        if distance is not None:
            nonconformity_scores = nonconformity_scores / distance
        
        # Check if nonconformity scores fall within the quantile threshold
        intervals = nonconformity_scores <= self.CP_quantiles
        return intervals