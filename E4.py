import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#Importing the Dataset
df = pd.read_csv('/content/car_data.csv')
df

#Histogram
sns.histplot(x = 'Fuel Type', data = df)

#Quartile
sns.boxplot(data = df.loc[:, ['Selling Price', 'Kilometers Driven']])

#Distribution Chart
sns.histplot(x = 'Year', data = df, kde = True)

#Scatterplot
sns.set_style('darkgrid')
sns.set(font_scale = 1)

sns.scatterplot(x = 'Year', y = 'Kilometers Driven', data = df)
plt.xlabel('Year')
plt.ylabel('Kilometers Driven')

#Scatter Matrix
sns.pairplot(df)
plt.show()

#Scatter Multiple
from matplotlib.pyplot import figure
plt.style.use('ggplot')

kilo_driven = df['Kilometers Driven']
sell_price = df['Selling Price']

plt.title('Relation between Kilometers Driven and Selling Price')
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price')
plt.scatter(x = kilo_driven[:500], y = sell_price[:500], c = 'green')

plt.tight_layout()
plt.show()

#Bubble Chart

#Define the variables you want to use for the x, y and size axes
x_var = 'Owner'
y_var = 'Car Condition'
size_var = 'Owner'

#Define the colors you want to use for the bubbles
colors = ['red', 'green', 'blue']

#Create the bubble chart using matplotlib
fig, ax = plt.subplots()

for i, color in enumerate(colors):
  subset = df[df['Owner'] == i]
  ax.scatter(subset[x_var], subset[y_var], s = subset[size_var]*100, c = color, alpha = 0.5, label = f'Owner {i+1}')

#Add Axis label and Legend
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)
ax.legend()

plt.show()

#Density Chart
df.Owner.plot.density(color = 'blue')
plt.title('Density Plot for Owner Type')
plt.show()

#Parallel Chart
import plotly.express as px
dimensions = ['Owner', 'Car Condition', 'Year']

fig = px.parallel_coordinates(df, dimensions = dimensions, color = 'Selling Price')
fig.show()

#Deviation Chart
sns.set(style = 'whitegrid')
sns.lineplot(x  = 'Owner', y = 'Car Condition', hue = 'Fuel Type', err_style = 'bars', data = df)

#Andrews Curves (1)
from pandas.plotting import andrews_curves
cols = ['Owner', 'Car Condition', 'Year']
andrews_curves(df[cols], 'Car Condition')
plt.show()

#Andrews Curves (2)
df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/'
'pandas/main/pandas/tests/io/data/csv/iris.csv')
pd.plotting.andrews_curves(df, 'Name')

