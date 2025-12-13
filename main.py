# Import Data Manipulation Libraries

import numpy as np 
import pandas as pd 

# Import Data Visulization Libraries

import seaborn as sns
import matplotlib.pyplot as plt 

# Import Data Warnings Libraries
import Warnings
Warnings.filterwarnings(action = 'ignore')


# Import Scikit Learn Libraries
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# step1: Data Ingestion()

def data_ingestion():
    pass

# step2: Data Exploration

def data_exploration(df):
    return num_stats,cat_stats,data_info

# step3: Data Preprocessing()

def data_preprocessing(df):
    return X_train,X_test,y_train,y_test = data_preprocessing(df)






# Function Calling
# step 1: Data Ingestion 
df = data_ingestion()

# step 2: Checking Descriptive Stats
num_stats,cat_stats,data_info = data_exploration(df)

# Step 3: Data Preprocessing

X_train,X_test,y_train,y_test = data_preprocessing(df)

