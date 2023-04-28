import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#Importing the Dataset
df = pd.read_csv('/content/car_data.csv')
df

#View Dimensions of Dataset
df.shape

#Preview the Dataset
df.head()

#View Summary of Dataset
df.info()

#Check for missing values
df.isnull().sum()

df.describe()

#Mean
df['Selling Price'].mean()

#Median
df['Selling Price'].median()

#Mode
df['Selling Price'].mode()

#Minimum Value
df['Selling Price'].min()

#Maximum Value
df['Selling Price'].max()

#Range
df['Selling Price'].max() - df['Selling Price'].min()

#Variance
round(df['Selling Price'].var(), 3)

#Standard Deviation
df['Selling Price'].std()

#Skewness
df['Selling Price'].skew()

#Kurtosis
df['Selling Price'].kurt()