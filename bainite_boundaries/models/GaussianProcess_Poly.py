import torch
import gpytorch
import numpy as np
from bainite_boundaries.models.GaussianProcess import ExactGPModel
from gpytorch.models import ApproximateGP

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from sklearn.preprocessing import StandardScaler

class ExactGPModel_Poly(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, num_epochs=501, lr=0.01, range_scale=1.0):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.lr = lr
        self.range_scale = range_scale

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x, train_y):

        likelihood = self.likelihood

        train_x = self.input_scaler.fit_transform(train_x)
        train_y = self.output_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()

        # Convert to PyTorch tensors
        train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PolynomialKernel(power=2)
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
            if i % 10 == 0:
                print(f'Iter {i} - Loss: {loss}  outputscale:{self.covar_module.outputscale}   noise: {self.likelihood.noise}' )

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



class VarGPModel_Poly_mixeddata(ApproximateGP):
    def __init__(self, train_x, train_y, likelihood, num_epochs=501, lr=0.01, range_scale=1.0,datatype=None):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution,learn_inducing_locations=True)
        

        
        super(VarGPModel_Poly_mixeddata,self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PolynomialKernel(power=2)
        )
        
        self.likelihood=likelihood
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.lr = lr
        self.range_scale = range_scale
        self.datatype=datatype

    def forward(self, x):
        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x, train_y):
        #if self.datatype==None:
        #    super().fit(train_x,train_y)
        
        likelihood = self.likelihood

        train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        



        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.parameters()}

        ], lr=self.lr)

        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        mll = gpytorch.mlls.VariationalELBO(likelihood, self,num_data=train_y.size(0))
        for i in range(self.num_epochs):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print(f'Iter {i} - Loss: {loss}   lengthscale:{self.covar_module.base_kernel.lengthscale}   noise: {self.likelihood.noise}' )

        self.likelihood = likelihood
        self.eval()

    def predict(self, test_x):
        test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
        
        self.eval()
        likelihood = self.likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(self(test_x))

        observed_pred = observed_pred.mean.cpu().numpy()
        return observed_pred
    

    def quantile_scores(self, calib_x, calib_y, coverage=0.8, distance=None):
        

        calib_x = torch.tensor(calib_x, dtype=torch.float32).to(self.device)
        calib_y = torch.tensor(calib_y, dtype=torch.float32).to(self.device)
        
        self.eval()
        likelihood = self.likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            validation_pred = likelihood(self(calib_x))

        y_calib_pred = validation_pred.mean.cpu().numpy()
        sigma_calib = validation_pred.stddev.cpu().numpy()
        lower_calib, upper_calib = validation_pred.confidence_region()
        lower_calib = lower_calib.cpu().numpy() 
        upper_calib = upper_calib.cpu().numpy()
        
        calib_y = calib_y.cpu().numpy()
        
        if distance is not None:
            nonconformity_scores = np.abs(calib_y - y_calib_pred) / (sigma_calib)
            quantile_scores = np.quantile(nonconformity_scores, coverage)
        else:
            # z_score_dict = {0.6: 0.253, 
            #                 0.7: 0.524, 
            #                 0.8: 0.842, 
            #                 0.9: 1.282, 
            #                 0.95: 1.645}
            z_score_dict = {0.7: 1.04, 0.9: 1.645}
            quantile_scores = z_score_dict[coverage] * sigma_calib
        
        self.CP_quantiles = quantile_scores
            

    def predict_interval(self, test_x, coverage=0.9, distance=None):
        
        
        test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
        self.eval()
        likelihood = self.likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(self(test_x))

        y_pred = observed_pred.mean.cpu().numpy()
        sigma = observed_pred.stddev.cpu().numpy()
        lower, upper = observed_pred.confidence_region()
        
        
        if distance is not None:
            sigma_test_CP = sigma * self.CP_quantiles 
            sigma_test_CP = sigma_test_CP
        else:
            z_score_dict = {0.7: 1.04, 0.9: 1.645}
            sigma_test_CP = z_score_dict[coverage] * sigma        
        
        return sigma_test_CP
