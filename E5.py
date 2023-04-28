import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#Importing the Dataset
df = pd.read_csv('/content/car_data.csv')
df

from sklearn.linear_model import LinearRegression
X = df[['Kilometers Driven']]
y = df[['Selling Price']]

lm = LinearRegression()
lm.fit(X, y)

y_pred = lm.predict(X)
print(y_pred)

#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(X,y_pred)
print("Mean Absolute Error: ",mae)

#Root Mean Square Error
from sklearn.metrics import mean_squared_error
import math
mse = mean_squared_error(X, y_pred)
rmse = math.sqrt(mse)
print("Root Mean Square Error: ",rmse)

#Mean Absolute Percentage Error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#Choose features and Target variable
X = df[['Year', 'Kilometers Driven']]
y = df['Selling Price']
lm = LinearRegression()
lm.fit(X,y)

y_pred = lm.predict(X)
print(y_pred)

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

#Train Machine Learning Model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions on testing set
y_pred = model.predict(X_test)

#Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("Mean Absolute Percentage Error: ", mape)

#Mean Absolute Scaled Error
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#Choose Target Variable
y = df['Selling Price']

#Split data into training and testing sets
train_size = int(len(y) * 0.8)
train, test = y[0:train_size], y[train_size: len(y)]

#Calculate Naive Forecast for Testing set
naive_forecast = test.shift(1).bfill()

#Calculate MAE
maenaive = np.mean(abs(test - naive_forecast))
print("MAE Naive: ",maenaive)

#Mean Absolute Scaled Error
MASE = mae / maenaive
print("Mean Absolute Scaled Error: ", MASE)