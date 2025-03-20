# try out different datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn import linear_model

def Calc_mol_per(weight_percentage):
    #weight_percentage=np.array([0,C,Si,Mn,Cr,V,Co,Al,N,B,P,Ti,S,W,Cu,Mo,Ni,Nb])
    mm=np.array([12.011,28.0855 ,54.9380 ,51.996 ,50.9415,58.933,26.98154 ,
                 14.007 ,10.811,30.974, 47.88, 32.066,183.85, 63.546,95.94,58.69  ,92.9064])
    mmFe=55.847

    
    
    if len(np.shape(weight_percentage))==2:
        mol_per=np.zeros(np.shape(weight_percentage))
        for k in range(np.shape(weight_percentage)[0]):
            w_perFe=100-np.sum(weight_percentage[k,:])    
            mol_per[k,:]=weight_percentage[k,:]/mm/(np.sum(weight_percentage[k,:]/mm)+w_perFe/mmFe)*100
    else:
        w_perFe=100-np.sum(weight_percentage)    
        mol_per=weight_percentage/mm/(np.sum(weight_percentage/mm)+w_perFe/mmFe)*100
    
    return mol_per
