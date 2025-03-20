from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

class LGBMRegressorModel:
    def __init__(self, n_estimators=500, max_depth=20, random_state=1, monotone_constraints=None):
        self.model = LGBMRegressor(n_estimators=n_estimators, 
                                   max_depth=max_depth, 
                                   random_state=random_state, 
                                   monotone_constraints=monotone_constraints, 
                                   verbose=-1)
        self.best_params_ = None
    
    def fit(self, train_x, train_y):
        
        # param_grid = {
        #     'n_estimators': [100],
        #     'max_depth': [10, 20, 50],
        #     'learning_rate': [0.01],
        #     'num_leaves': [31, 50, 100],
        # }
        
        # self.tune_hyperparameters(train_x, train_y, param_grid)
        

        self.model.fit(train_x, train_y)
        
    def predict(self, test_x):
        return self.model.predict(test_x)

    
    def tune_hyperparameters(self, train_x, train_y, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
        """Tune hyperparameters using GridSearchCV"""
        print("Tuning hyperparameters...")
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                #    scoring=scoring,
                                   cv=cv,
                                   n_jobs=n_jobs,
                                   verbose=0)
        grid_search.fit(train_x, train_y)
        self.best_params_ = grid_search.best_params_
        # Update the model with the best parameters
        print("Best hyperparameters found: ", self.best_params_)
        self.model = grid_search.best_estimator_

    def get_best_params(self):
        """Get the best hyperparameters found by GridSearchCV"""
        return self.best_params_
