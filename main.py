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


# Step 1:  Data Ingestion()

def data_ingestion():
    return pd.read_csv(filepath)

# Step2: Data Exploration 

def data_exploration(df):
    
    from collections import OrderedDict
    
    # Numerical Stats:
    numerical_col = df.select_dtypes(exclude = 'object').columns
    categorical_col = df.select_dtypes(include = 'object').columns
    
    num_stats = []
    cat_stats = []
    data_info = []
    
    for i in numerical_col:
        
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        LW = Q1-1.5*IQR
        UW = Q3+1.5*IQR
        
        Outlier_Count = len(df[(df[i] < LW) | (df[i] > UW)])
        Outlier_Percent = Outlier_Count / len(df) * 100

        numerical_stats = OrderedDict({
            "Feature":i,
            "Count":df[i].count(),
            "Outlier_Count": Outlier_Count,
            "Outlier_Percent":Outlier_Percent
        })
        num_stats.append(numerical_stats)
        numerical_stats_report = pd.DataFrame(num_stats)
        

    for i in categorical_col:
        categorical_stats = OrderedDict({
            "Feature":i,
            "Count":df[i].count(),
            "Unique Values":df[i].unique(),
            "Number of Unique Values":df[i].nunique(),
            "Value_Counts": df[i].value_counts()
        })
        cat_stats.append(categorical_stats)
        categorical_stats_report = pd.DataFrame(cat_stats)
    
    for i in df.columns:
        data_information = OrderedDict({
            "Feature":i,
            "Data Types":df[i].dtype,
            "ValueCount":df[i].count()
        })
        data_info.append(data_information)
        
        data_stats_report = pd.DataFrame(data_info)
    
    
    return numerical_stats_report,categorical_stats_report,data_stats_report
    
    
        
# Step3: Data Preprocessing 
def data_preprocessing(df):
    
    # Split the dataset into X and y
    X = df.drop(columns = 'Performance Index',axis = 1)
    y = df['Performance Index']
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10)


    # Encoding
    categorical_col = X.select_dtypes(include="object").columns
    for i in categorical_col:
        le = LabelEncoder()
        X_train[i] = le.fit_transform(X_train[i])
        X_test[i] = le.transform(X_test[i])


    # Scaling
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    return X_train, X_test, y_train, y_test
    
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Step4: Model Building
def model_build(df):
    
    model_comparison = []

    models = {
        "Linear Regression": LinearRegression(),
        "DecisionTree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
    }

    for model_name,model in models.items():
       model.fit(X_train,y_train)
       y_pred = model.predict(X_test)

       r2score = r2_score(y_test,y_pred)

       model_comparison.append({"model_name":model,
                              "R2 Score":r2score})
       DataFrame = pd.DataFrame(model_comparison)
    return DataFrame                          
# Function Calling :

df = data_ingestion()

numerical_stats_report,categorical_stats_report,data_stats_report = data_exploration(df)

X_train,X_test,y_train,y_test = data_preprocessing(df)

LinearRegression_models_report=model_build(df)


# Test
# print(df)
#print(numerical_stats_report)
# print(data_stats_report)
# print(X_test.shape)
print(LinearRegression_models_report)