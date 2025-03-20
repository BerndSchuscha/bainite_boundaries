from xgboost import XGBRegressor
import numpy as np

  

class XGBRegressorModel:
    def __init__(self, n_estimators=500, max_depth=50, random_state=1, monotone_constraints=None, transform_features_flag=False):
        self.model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, monotone_constraints=monotone_constraints, verbosity=0)
        self.transform_features_flag = transform_features_flag
        
    def transform_features(self, x):
        feature = x[:, -1:]
        x_transformed = x
        for i in [2,4,6]:
            x_transformed = np.hstack([x_transformed, np.log(feature)** i])
            
        x_transformed = np.hstack([x_transformed, np.log(feature)])
        return x_transformed
    
    
    def fit(self, train_x, train_y):
        if self.transform_features_flag:
            train_x = self.transform_features(train_x)
        self.model.fit(train_x, train_y)
    
    def predict(self, test_x):
        if self.transform_features_flag:
            test_x = self.transform_features(test_x)
        return self.model.predict(test_x)
