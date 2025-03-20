import torch
import gpytorch
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class ExactGPModel_linearmean(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_epochs=501, lr=0.01, range_scale=1.0):
        super(ExactGPModel_linearmean, self).__init__(train_x, train_y, likelihood)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.lr = lr
        self.range_scale = range_scale

        # Initialize scalers for input and output
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x, train_y):
        # Scale inputs and outputs
        train_x = self.input_scaler.fit_transform(train_x)
        train_y = self.output_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
        # Convert to PyTorch tensors
        train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)

        likelihood = self.likelihood
        super(ExactGPModel_linearmean, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.LinearMean(input_size=train_x.shape[1])
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(self.num_epochs):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print(f'Iter {i} - Loss: {loss}   lengthscale: {self.covar_module.base_kernel.lengthscale}   noise: {self.likelihood.noise}')

        self.likelihood = likelihood
        self.eval()

    def predict(self, test_x):
        # Scale inputs
        test_x = self.input_scaler.transform(test_x)
        test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)

        self.eval()
        likelihood = self.likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(self(test_x))

        # Inverse-transform predictions
        pred_mean = self.output_scaler.inverse_transform(observed_pred.mean.cpu().numpy().reshape(-1, 1)).flatten()
        return pred_mean

    def predict_uncertainty(self, test_x, alpha=0.05):
        # Scale inputs
        test_x = self.input_scaler.transform(test_x)
        test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)

        self.eval()
        likelihood = self.likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(self(test_x))
            variance = observed_pred.variance.cpu().numpy()

        # Scale uncertainty back to original output scale
        scale_factor = self.output_scaler.scale_[0]
        stddev = np.sqrt(variance) * scale_factor
        return stddev



    def save_model(self,path):
        joblib.dump(self.input_scaler , path+'input_scaler.pkl')
        joblib.dump(self.output_scaler , path+'output_scaler.pkl')


    def load_model(self,path):
        self.input_scaler = joblib.load(path+'input_scaler.pkl')
        self.output_scaler = joblib.load(path+'output_scaler.pkl')
