
# import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import bainite_boundaries

from bainite_boundaries.utils.MS_AL import Calc_mol_per    

def load_Bainite():
    data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data', 't90_data.csv')
    df = pd.read_csv(data_path, delimiter=',')
    df = df.drop_duplicates()

    # Drop the first column (assumed as index or unnecessary)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[1], axis=1)
    df.drop(columns=['N','B','P','TI','S','W','NB'], inplace=True)
    print(df.head())
    
    
    # Group by 'Nr' (if 'Nr' column still exists after dropping the first column)
    # Replace 'Nr' with the correct name if it's different in your actual data
    grouped_data = {}
    x = []
    y = []
    for name, group in df.groupby("Nr"):
        x.append(np.array( group.iloc[:, 1:-1].values.tolist()) )
        y.append(np.array( group.iloc[:, -1].values.tolist()) )

    elements_vector = df.columns[1:-1]
    return x, y, list(elements_vector)

def load_Ferrite():
    data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data', 'Ferrite.txt')
    df = pd.read_csv(data_path, sep=';')  # loads dataset in a pandas DataFrame object
    # drop duplicates
    print(df.head())
    # replace nan values with 0
    df.fillna(0, inplace=True)
    df = df.drop_duplicates()
    df.drop(columns=['N','P','Ti','S','Nb'], inplace=True)
    print(df.head())

    grouped_data = {}
    x = []
    y = []

    # get index of the columns with the name Atemp
    Atemp_idx = df.columns.get_loc('Atemp')
    T_idx = df.columns.get_loc('T')
    # ASTM_idx = df.columns.get_loc('ASTM Grain Size')
    y_idx = df.columns.get_loc('t')

    for name, group in df.groupby("ID"):
        x_i = np.array( group.iloc[:, 1:Atemp_idx].values.tolist()) 
        x_i = np.append(x_i, group.iloc[:, T_idx].values[:, None], axis=1)

        if np.isnan(x_i).any():
            print("There are nan values in x_i")
        
        # x_i = np.append(x_i, group.iloc[:, ASTM_idx].values[:, None], axis=1)
        x.append( x_i )
        y.append( np.array( group.iloc[:, y_idx].values.tolist()) )
    elements_vector = df.columns[1:Atemp_idx].tolist()
    return x, y, list(elements_vector)

def load_Ferrite_critCR():
    data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data', 'Ferrite_critCR.csv')
    data = pd.read_csv(data_path, delimiter=';')
    data = data.drop(data.columns[0], axis=1)
    print(data.head())
    Atemp_idx = data.columns.get_loc('Atemp')
    x = data.iloc[:, 0:Atemp_idx].values
    y = data.iloc[:, -1].values
    features = list(data.columns[0:Atemp_idx])
    return x, y, features
    

def load_Austensite():
    
    # print the current working directory    
    data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data')
    x = np.loadtxt(data_path / 'Austenite_input2.txt')
    y = np.loadtxt(data_path / 'Austenite_y2.txt')

    # remove any nan values
    nan_indices = np.argwhere(np.isnan(y))
    # print("Number of nan values: ", len(nan_indices))
    # exchange the nan values with 1
    y[nan_indices] = -1000

    # check if there are any nan values 
    nan_indices = np.argwhere(np.isnan(y))
    # print("Number of nan values: ", len(nan_indices))

    # remove -1 values of y 
    neg_indices = np.argwhere(y == -1)
    # print("Number of -1 values: ", len(neg_indices))
    x = np.delete(x, neg_indices, axis=0)
    y = np.delete(y, neg_indices, axis=0)
    
        
    y_class = np.zeros_like(y)
    y_class[y > 0] = 1
    y_class[y < 0] = 0
    
    constraint_violation = np.mean(y_class==0) * 100
    print(f'Constraint violation: {constraint_violation:.2f}%')
    features = ['C', 'SI', 'MN', 'CR', 'AL', 'MO', 'V']
    
    return x, y_class, features

def load_final():
    
    # print the current working directory    
    data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data')
    x = np.loadtxt(data_path / 'final_samples.txt')
    y = np.random.rand(np.shape(x)[0]) # given random vector for filling
    features = ['C', 'SI', 'MN', 'CR', 'AL', 'V', 'Mo','T','T_crit']
    
    return x, y, features

def load_Martensite_start():
    data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data', 'MS_qilu.txt')
    
    df = pd.read_csv(data_path, sep='\t')  # loads dataset in a pandas DataFrame object
    df.fillna(0, inplace=True)
    
    y = df['Ms (K)'].values-273 #'TAUST (K)' 'Ms (K)'
    excluded = ['Ms (K)', 'TAUST (K)', 'Source']
    X = df.drop(excluded, axis=1)
    #df.drop(columns=['N','B','P','TI','S','NB'], inplace=True)
    X=X.to_numpy()
    print(np.shape(X))
    x=X#x=Calc_mol_per(X)    
    
    features = list(df.columns)
    print(features)
    excluded = ['Ms (K)', 'TAUST (K)', 'Source','N','B','P','TI','S','NB' ]
    x=np.delete(x,[7,8,9,10,11,16],axis=1)
    # remove TAUST (K) and Ms (K) from the features
    features = [f for f in features if f not in excluded]
    return x, y, features

def load_Martensite_start_RA():
    data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data', 'MS_qilu_RA.txt')
    df = pd.read_csv(data_path, sep='\t')  # loads dataset in a pandas DataFrame object
    df.fillna(0, inplace=True)
    
    y = df['Ms (K)'].values-273 #'TAUST (K)' 'Ms (K)'
    excluded = ['Ms (K)', 'TAUST (K)', 'Source','Tiso (°C)']
    X = df.drop(excluded, axis=1)
    #df.drop(columns=['N','B','P','TI','S','NB'], inplace=True)
    X=X.to_numpy()
    print(np.shape(X))
    x=X#x=Calc_mol_per(X)    
    
    features = list(df.columns)
    print(features)
    excluded = ['Ms (K)', 'TAUST (K)', 'Source','N','B','P','TI','S','NB','Tiso (°C)' ]
    x=np.delete(x,[7,8,9,10,11,16,-1],axis=1)
    # remove TAUST (K) and Ms (K) from the features
    features = [f for f in features if f not in excluded]
    return x, y, features

def load_Bainite_start():
    data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data', 'Bs_data.csv')
    data = pd.read_csv(data_path, delimiter=',')
    x = data.iloc[:, 2:-1].values    
    y = data.iloc[:, -1].values
    # get column names
    features = list(data.columns[2:-1])
    # remove /% from the column names
    features = [f.split(' /')[0] for f in features]
    return x, y, features

def load_dataset(which_data,base_problem=None):
    
    print(f'Loading {which_data} data')
    if which_data == 'Bainite':
        x, y, features = load_Bainite()
    elif which_data == 'Ferrite':
        x, y, features = load_Ferrite()
    elif which_data == 'Austensite':
        x, y, features = load_Austensite()
    elif which_data == 'Martensite_start':
        x, y, features = load_Martensite_start()
    elif which_data == 'Martensite_start_RA':
        x, y, features = load_Martensite_start_RA()
    elif which_data == 'Bainite_start':
        x, y, features = load_Bainite_start()
    elif which_data == 'Ferrite_critCR':
        x, y, features = load_Ferrite_critCR()
    elif which_data == 'final':
        x, y, features = load_final()
        if base_problem=='Martensite_start_RA':

            data_path = Path(str(bainite_boundaries.PROJECT_ROOT),'bainite_boundaries', 'data', 'cRA_final.txt')
            df = pd.read_csv(data_path, sep='\t')  # loads dataset in a pandas DataFrame object
            x[:,0]=df['CRA'].to_numpy()
    # if x is an array, show the shape
    if isinstance(x, np.ndarray):
        print(f'x shape: {x.shape}')
        print(f'y shape: {y.shape}')
    else:
        print(f'List of x, length: {len(x)}')
        print(f'Shape of the first element in x: {x[0].shape}')
    
    features = [f.lower() for f in features]
    return x, y, features