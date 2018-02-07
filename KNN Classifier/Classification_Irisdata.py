# -*- coding: utf-8 -*-
"""
@author   : Hima Bindu Sigili
StudentID : 801023234

"""
#Using pandas library to load the Iris dataset 
import pandas as pd

'''Download Iris dataset from the official website 
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
Import the downloaded Iris dataset into excel and rename columns 
as 4 Iris features(sepal length,sepal width,petal length,petal width)
and last column as target and save it as .csv file'''

data1= pd.read_csv('C:/Users/bindu/Downloads/irisdata.csv')
data1.shape

#.head() function gives first 5 samples of data
data1.head()

# To apply scikit learn ML techniques the features should be in 2D matrix 
#and response should be a 1D array
#so use .as_matrix method to convert the first 4 columns data as matrix
X = data1[['sepal length',
           'sepal width','petal length','petal width']].as_matrix()
print(X.shape)

#Also the features and target should be of same datatype 
#so use user defined map to convert string values into float
x={'Iris-setosa':0.0,'Iris-versicolor':1.0,'Iris-virginica':2.0}
data1['target']=data1['target'].map(x)

y= data1['target'].values
print(y.shape)

#check the dataframe types the features and target must be of same data type
data1.dtypes

#Use Scikit learn to train the neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
kno= KNeighborsClassifier(n_neighbors=5)
print(kno)

#Train the model 
kno.fit(X,y)

#Predict the output for out of sample data
kno.predict([[3,5,4,2]])
X_new=[[3,5,4,2],[5,4,3,2]]
kno.predict(X_new)


#alternatively we can use Iris dataset from scikit learn sample datasets
# to predict the output
'''
from sklearn.datasets import load_iris
iris= load_iris()
X=iris.data
y=iris.target
print(X.shape)
print (y.shape)
type(iris)
from sklearn.neighbors import KNeighborsClassifier
kno= KNeighborsClassifier(n_neighbors=5)
print(kno)
kno.fit(X,y)
X_new=[[3,5,4,2],[5,4,3,2]]
kno.predict(X_new)
kno.predict([[3,5,4,2]])
'''


#Part of the following code(only plotting part) has been downloaded 
#from the Scikit learn website examples
#for neighbor classifiers to plot the boundary regions. URL link:
"""
http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py 
""" # noqa

#some changes are done like instead of sepal measurements,I used petal
# measurements as X and Y axis in the plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
n_neighbors = 5

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset (note- I used last two features here)
#X = X[:, [0,1]]  '''for sepal length and width'''
X = X[:, [2,3]]

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

kno.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = kno.predict(np.c_[xx.ravel(), yy.ravel()])

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
plt.title("3-Class classification (k = %i)"
          % (n_neighbors))

plt.show()