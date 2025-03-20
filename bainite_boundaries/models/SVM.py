from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

class SVC_efficient:
    def __init__(self):
        self.feature_map_nystroem = None
        self.model = None
        self.platt_scaler = None

    def fit(self, train_x, train_y, cv=5):
        # Split data into training and calibration sets
        train_x, calibrate_x, train_y, calibrate_y = train_test_split(
            train_x, train_y, test_size=0.1, random_state=1, stratify=train_y
        )
        
        # Compute gamma for the RBF kernel approximation
        gamma = 1 / (train_x.shape[1] * np.var(train_x))

        # Set up the Nystroem feature map
        n_components = 300  # Tunable parameter
        self.feature_map_nystroem = Nystroem(gamma=gamma, random_state=1, n_components=n_components)
        data_transformed = self.feature_map_nystroem.fit_transform(train_x)
        calibration_transformed = self.feature_map_nystroem.transform(calibrate_x)

        # Train an SGDClassifier
        self.model = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, max_iter=1000, tol=1e-3)
        self.model.fit(data_transformed, train_y)

        # Apply Platt Scaling using Logistic Regression
        decision_values = self.model.decision_function(calibration_transformed)
        self.platt_scaler = LogisticRegression()
        self.platt_scaler.fit(decision_values.reshape(-1, 1), calibrate_y)

    def predict(self, test_x):
        # Transform test data and use SGDClassifier for predictions
        data_transformed = self.feature_map_nystroem.transform(test_x)
        return self.model.predict(data_transformed)

    def predict_proba(self, test_x):
        # Transform test data and compute decision values
        data_transformed = self.feature_map_nystroem.transform(test_x)
        decision_values = self.model.decision_function(data_transformed)
        # Use Platt Scaler to get probabilities
        return self.platt_scaler.predict_proba(decision_values.reshape(-1, 1))
