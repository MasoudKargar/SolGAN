import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

data = pd.read_csv("new2.csv")

dataframe = pd.read_csv("new2.csv", header=None)
#dataframe = pandas.read_csv("new3_gan_287.csv", header=None)
dataset=dataframe.values
# split into input (X) and output (Y) variables
x = dataset[:,0:3]
y = dataset[:,3]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

linear_regression = LinearRegression()
linear_regression.fit(xtrain, ytrain)
predictions = linear_regression.predict(xtest)

# Calculation of R2 Score
from sklearn.model_selection import cross_val_score
print(cross_val_score(linear_regression, x, y, cv=10, scoring="r2").mean())