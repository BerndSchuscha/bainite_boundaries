import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import monotonicnetworks as lmn
import tqdm
from sklearn.preprocessing import StandardScaler
class Monoton_MLPRegressorTorch(nn.Module):
    def __init__(self, input_size, hidden_size, random_state):
        super(Monoton_MLPRegressorTorch, self).__init__()
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = torch.nn.Sequential(
            lmn.direct_norm(torch.nn.Linear(input_size, hidden_size), kind="one-inf"),
            lmn.GroupSort(2),
            lmn.direct_norm(torch.nn.Linear(hidden_size, hidden_size), kind="inf"),
            lmn.GroupSort(2),
            lmn.direct_norm(torch.nn.Linear(hidden_size, 1), kind="inf"),
        )
        
    def forward(self, x):
        return self.model(x)


class Monoton_NeuralNetworkModelPytorch:
    def __init__(self, hidden_size, apply_pca=False, explained_variance=0.999, transformations=None, random_state=1,monotonicty=[1]):
        self.model = None
        self.hidden_size = hidden_size
        self.apply_pca = apply_pca
        self.explained_variance = explained_variance
        self.transformations = transformations
        self.random_state = random_state
        self.monotonicty=monotonicty

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
                
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

    def fit(self, X, y, learning_rate=0.001, alpha=0.01, max_iter=3000, early_stopping=True, val_fraction=0.1, n_iter_no_change=10):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)



        print('X1',np.shape(X))
        if self.apply_pca and self.pca is not None:
            X = self.pca.transform(X)
        if self.transform_features_flag: 
            X = self.transform_features(X)  
        print('X2',np.shape(X))

        X = self.input_scaler.fit_transform(X)
        y = self.output_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        print(self.monotonicty)

        self.model = lmn.MonotonicWrapper(Monoton_MLPRegressorTorch(np.shape(X)[1], hidden_size=self.hidden_size, random_state=self.random_state),monotonic_constraints=self.monotonicty)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_fraction, random_state=self.random_state)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        best_loss = np.inf
        no_improve_epochs = 0

        for epoch in range(max_iter):
            self.model.train()
            optimizer.zero_grad()
            
            predictions = self.model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            # Validation loss for early stopping
            if epoch % 10 ==0:
                print(f"epoch: {epoch} loss: {loss.item():.4f}")
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val)
                    val_loss = criterion(val_predictions, y_val)
                
                if val_loss < best_loss:
                    # print(f"Epoch {epoch}: Improved validation loss: {val_loss:.4f}")
                    best_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    
                if early_stopping and no_improve_epochs >= n_iter_no_change:
                    print(f"Early stopping at epoch {epoch}")
                    break

            scheduler.step()
        
        print("Training complete.")

    def predict(self, X):
        
        if self.apply_pca and self.pca is not None:
            X = self.pca.transform(X)
        if self.transform_features_flag: 
            X = self.transform_features(X)
        X = self.input_scaler.transform(X)
        if self.model is None:
            raise ValueError("The model has not been trained yet. Call 'fit' before 'predict'.")
        
         # Set model to evaluation mode
        with torch.no_grad():
            self.model.eval() 
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.model(X_tensor)
        predictions = self.output_scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).flatten()
        return predictions

