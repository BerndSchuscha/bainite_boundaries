from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

class RandomForestRegressorModel:
    def __init__(self, n_estimators=100, max_depth=20, random_state=1, transformations=None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.model = RandomForestRegressor()
        self.best_params_ = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
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
    
    def fit(self, train_x, train_y):
        param_grid = {
            'n_estimators': [200],
            'max_depth': [10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4],
            # 'max_features': ['auto', 'sqrt', 'log2']
        }
        # self.tune_hyperparameters(train_x, train_y, param_grid)
        # print("Fitting the model with the best hyperparameters...")
        # print("Best Parameters:", self.best_params_)
        # self.model = RandomForestRegressor(**self.best_params_, random_state=self.model.random_state)
        if self.transform_features_flag: 
            train_x = self.transform_features(train_x)
        
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.model.fit(train_x, train_y)
        
    def predict(self, test_x):
        if self.transform_features_flag:
            test_x = self.transform_features(test_x)
        return self.model.predict(test_x)
 
    def tune_hyperparameters(self, train_x, train_y, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
        """Tune hyperparameters using GridSearchCV"""
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   scoring=scoring,
                                   cv=cv,
                                   n_jobs=n_jobs,
                                   verbose=1)
        grid_search.fit(train_x, train_y)
        self.best_params_ = grid_search.best_params_
        print("Best Parameters:", self.best_params_)

    def get_best_params(self):
        """Get the best hyperparameters found by GridSearchCV"""
        return self.best_params_
