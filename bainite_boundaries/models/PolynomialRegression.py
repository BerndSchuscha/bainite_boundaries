from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.optimize import minimize

class PolynomialRegressionModel:
    def __init__(self, degree=2, apply_pca=False, explained_variance=0.999, random_state=1, bandwidth=0.1):
        self.degree = degree
        self.apply_pca = apply_pca
        self.explained_variance = explained_variance
        self.random_state = random_state
        self.model = Ridge()
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.pca = None
        self.bandwidth = bandwidth  # for KDE
    
    def fit(self, train_x, train_y):
        # Transform the data to polynomial features
        train_x_poly = self.poly.fit_transform(train_x)
        
        # Apply PCA if required
        if self.apply_pca:
            self.pca = PCA(self.explained_variance, random_state=self.random_state)
            train_x_poly = self.pca.fit_transform(train_x_poly)
        
        # Fit the Ridge regression model
        self.model.fit(train_x_poly, train_y)
        
    def predict(self, test_x):
        # Transform the test data to polynomial features
        test_x_poly = self.poly.transform(test_x)
        
        self.test_x = test_x
        self.test_x_poly = test_x_poly
        # Apply PCA if it was used in training
        if self.apply_pca and self.pca is not None:
            test_x_poly = self.pca.transform(test_x_poly)
        
        return self.model.predict(test_x_poly)
    
    

class PolynomialRegressionModel_mixeddata(PolynomialRegressionModel):
    def __init__(self, degree=2, apply_pca=False, explained_variance=0.999, 
                 random_state=1, bandwidth=0.1,datatype=None):
        
        super().__init__( degree, apply_pca, explained_variance, random_state, bandwidth)
        self.datatype=datatype
    def _loss_function(self, coef, X, y):
        y_pred = X @ coef

        diff=y - y_pred

        L=np.logical_and(self.datatype==True,diff<0)


        loss = ((diff) ** 2)

        loss[L]=0
        loss[np.logical_not(L)]=0

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