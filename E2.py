import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#Importing the Dataset
df = pd.read_csv('/content/car_data.csv')
df

df['Selling Price'].mean()

sampData = df['Selling Price'][np.argsort(np.random.random(2200))[:1000]]
sampData

meanSampData = sampData.mean()
hypmean = 418443.07
N = 2200
standPop = np.std(df['Selling Price'])

import math
z = abs((meanSampData - hypmean)/ (standPop/math.sqrt(N)))
print(z)
if(z<1.96):
  print('As calculated z score',z,'is less than 1.96 (tabular z score), we reject the null hypothesis')
else:
  print('As calculated z score',z,'is greater than 1.96 (tabular z score), we do not reject the null hypothesis')
