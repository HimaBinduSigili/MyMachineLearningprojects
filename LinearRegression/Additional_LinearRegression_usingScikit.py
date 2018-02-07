# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:02:21 2018

@author   : Hima Bindu Sigili
StudentID : 801023234
"""
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the iris dataset
iris = datasets.load_iris()
X=iris.data
y=iris.target

# Split the data and targets into training/testing sets
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(
        X, y, test_size=0.2, random_state=3)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(iris_X_train, iris_y_train)

# Make predictions using the testing set
iris_y_pred = regr.predict(iris_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(iris_y_test, iris_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(iris_y_test, iris_y_pred))

# Plot outputs
# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset (note- I used last two features here)
#X = X[:, [0,1]]  '''for sepal length and width'''
X_train = iris_X_train[:, [2,3]]
y_train=iris_y_train

h = .02  # step size in the mesh

regr.fit(X_train, y_train)

X = iris_X_test[:, [2,3]]
y = iris_y_test


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,h),
                     np.arange(y_min, y_max,h))
Z = regr.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.title("Linear Regression")

plt.show()