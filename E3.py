import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#Importing the Dataset
df = pd.read_csv('/content/car_data.csv')
df

#Complete Case Analysis:
df.shape

na_variables = [ var for var in df.columns if df[var].isnull().mean() > 0 ]
na_variables

data_na = df[na_variables].isnull().mean()
data_na

#Implementing the CCA techniques to remove Missing Data
data_cca = df.dropna(axis=0)
df.shape, data_cca.shape

#2. Abritrary Value Imputation
#USing Arbitrary Imputation technique, we will impute nan with "Not Available"
arb_impute = df['Insurance'].fillna('Not Available')
arb_impute.unique()

missing_data = df.isnull().mean()
missing_data

#3. Frequent Category Imputation
df['Insurance'].groupby(df['Insurance']).count()

# Expired has highest frequency. We can also verify it by checking the Mode
df['Insurance'].mode()

#Using Frequent Category Imputer
frq_impute = df['Insurance'].fillna('Expired')
frq_impute.unique()