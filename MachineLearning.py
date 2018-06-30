import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Let's model!
from sklearn.linear_model import LinearRegression as LinReg

plt.style.use('ggplot')
# Read in data into a dataframe
df = pd.read_csv('GlobalTemperatures.csv')

# Show the first 5 rows of the table
df.tail(10)
# Let's just consider the LandAverageTemperature
# "A primarily label-location based indexer"
df = df.ix[:,:2]
df.head()
df.describe()
# # Cursory plot
# plt.figure(figsize = (15, 5))
# plt.scatter(x = df['LandAverageTemperature'].index, y = df['LandAverageTemperature'])
# plt.title("Average Land Temperature 1750-2015")
# plt.xlabel("Year")
# plt.ylabel("Average Land Temperature")
# plt.show()

# Convert to datetime object
times = pd.DatetimeIndex(df['dt'])


# Use previous valid observation to fill gap
df['LandAverageTemperature'] = df['LandAverageTemperature'].fillna(method='ffill')


# Regroup and plot
grouped = df.groupby([times.year]).mean()

# Better, but still not perfect
# What are some other ways to fill the NaN values?
# plt.figure(figsize = (15, 5))
# plt.plot(grouped['LandAverageTemperature'])
# plt.show()

x = grouped.index.values.reshape(-1, 1)
y = grouped['LandAverageTemperature'].values

reg = LinReg()
reg.fit(x, y)
y_preds = reg.predict(x)
print("Accuracy: " + str(reg.score(x, y)))

plt.figure(figsize = (15, 5))
plt.title("Linear Regression")
plt.scatter(x = x, y = y_preds)
plt.scatter(x = x, y = y, c = "r")
plt.show()

prediction = reg.predict(2050)
print(prediction)

