# Import Data Manipulation Libraries
import numpy as np 
import pandas as pd 

# Import Data Visualization Libraries
import seaborn as sns 
import matplotlib.pyplot as plt 

# Import Data Warning Libraries
import warnings
warnings.filterwarnings(action = 'ignore')

# Import Scikit Learn Libraries
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

filepath = 'https://raw.githubusercontent.com/rushikeshpatil-123/StudentPerformanceModel_RegressionAnalysis/refs/heads/main/Data/Student_Performance.csv'


# step1: Data Ingestion()

def data_ingestion():
    return pd.read_csv(filepath)


# function Calling 
df = data_ingestion()

# Test 
print(df)