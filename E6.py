import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#Importing the Dataset
df = pd.read_csv('/content/car_data.csv')
df

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#Generate a sample imbalanced dataset
X, Y = make_classification(n_samples = 4600, n_features = 2, n_informative = 2, 
                          n_redundant = 0, n_repeated = 0, n_classes = 2,
                          n_clusters_per_class = 1,
                          weights = [0.95, 0.05],
                          class_sep = 0.8, random_state = 100)

X1, X2 = list(), list()
for i, j in enumerate(X):
  X1.append(j[0])
  X2.append(j[1])

df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y })

pal = sns.color_palette('tab10')
print(pal.as_hex())

#sns.set_patlette('icefire)
sns.set_palette(['#55a3cd', '#9c2f45'])
sns.palplot(sns.color_palette())

#Imbalanced Data
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6), dpi = 100)

sns.despine(left = True)
sns.scatterplot(x = 'X1', y = 'X2', hue = 'Y', data = df)

df['Y'].value_counts()

x = df.drop('Y', axis = 1)
y = df['Y']

from imblearn.over_sampling import SMOTE
from collections import Counter

counter = Counter(y)
print('Before', counter)

#Oversampling the train datset using SMOTE
smt = SMOTE()

X_train1, y_train1 = smt.fit_resample(x,y)

counter = Counter(y_train1)
print('After', counter)