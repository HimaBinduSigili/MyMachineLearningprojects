# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:29:24 2018

@author   : Hima Bindu Sigili
StudentID : 801023234
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Function to calculate error 
def calculate_error(c,m1,m2,m3,m4,X,y):
    totalError=0
    N = float(len(X))
    for i in range(0,len(X)):
        x1=X[i,0]
        x2=X[i,1]
        x3=X[i,2]
        x4=X[i,3]
        totalError += (y[i]-(m1*x1+m2*x2+m3*x3+m4*x4+c))**2
    return totalError/(N-2)

#This function uses Gradient descent algorithm and
#gives us the coefficients for 10,000 iterations in my case   
def gradient_descent(X, y, starting_c, starting_m1, starting_m2,
                            starting_m3, starting_m4, learning_rate,
                            num_iterations):
    c=starting_c
    m1=starting_m1
    m2=starting_m2
    m3=starting_m3
    m4=starting_m4
    
    for i in range(num_iterations):
        c,m1,m2,m3,m4 =step_gradient(c,m1,m2,m3,m4,X,y,learning_rate)
    return [c,m1,m2,m3,m4]

#calculate step gradient for each iteration at a learning rate of 0.0001
def step_gradient(c_current,m1_current,m2_current,m3_current,m4_current,X,y,
                  learning_rate):
    c_gradient = 0
    m1_gradient = 0
    m2_gradient = 0
    m3_gradient = 0
    m4_gradient = 0
    N = float(len(X))
    for i in range(0, len(X)):
         x1=X[i,0]
         x2=X[i,1]
         x3=X[i,2]
         x4=X[i,3] 
         c_gradient += -(2/N)*(y[i]-((m1_current * x1)+(m2_current * x2)+
                               (m3_current * x3)+(m4_current * x4)+c_current))
         m1_gradient += -(2/N)*x1*(y[i]-((m1_current * x1)+(m2_current * x2)+
                                   (m3_current * x3)+(m4_current * x4)
                                   +c_current))
         m2_gradient += -(2/N)*x2*(y[i]-((m1_current * x1)+(m2_current * x2)+
                                   (m3_current * x3)+(m4_current * x4)
                                   +c_current))
         m3_gradient += -(2/N)*x3*(y[i]-((m1_current * x1)+(m2_current * x2)+
                                   (m3_current * x3)+(m4_current * x4)
                                   +c_current))
         m4_gradient += -(2/N)*x4*(y[i]-((m1_current * x1)+(m2_current * x2)+
                                   (m3_current * x3)+(m4_current * x4)
                                   +c_current))
    new_c = c_current- (learning_rate*c_gradient)
    new_m1 = m1_current - (learning_rate*m1_gradient)
    new_m2 = m2_current - (learning_rate*m2_gradient)
    new_m3 = m3_current - (learning_rate*m3_gradient)
    new_m4 = m4_current - (learning_rate*m4_gradient)
    return [new_c, new_m1, new_m2, new_m3, new_m4]             

# Train the model 
def training(X_train,y_train):
    learning_rate = 0.0001
    initial_c = 0
    initial_m1 = 0
    initial_m2 = 0
    initial_m3 = 0
    initial_m4 = 0
    num_iterations = 10000
    print('starting gradient descent at c={0}, m1={1},m2={2},m3={3},m4={4},error={5}'
           .format(initial_c,initial_m1,initial_m2,initial_m3,initial_m4,
                   calculate_error(initial_c,initial_m1,initial_m2,initial_m3,
                                   initial_m4,X_train,y_train)))
    c, m1,m2,m3,m4 = gradient_descent(X_train,y_train,
                                             initial_c,initial_m1,initial_m2,
                                             initial_m3,initial_m4,
                                             learning_rate, num_iterations)
    print ('After {0} iterations ending point at c={1}, m1={2},m2={3},m3={4},m4={5}, error={6}'
           .format(num_iterations,c,m1,m2,m3,m4,
                   calculate_error(c,m1,m2,m3,m4,X_train,y_train)))
    return [c,m1,m2,m3,m4]

''' Download Iris dataset from the official website 
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
Import the downloaded Iris dataset into excel andsave it as .csv file'''

'''if __name__ == "__main__":
    points = np.genfromtxt('C:/Users/bindu/Downloads/ML/irisdata.csv',
                           delimiter = ',')
   # X=points[:,0:4]
    #y=points[:,4]
    np.random.shuffle(points)
    X_train = points[0:120,0:4]
    y_train = points[0:120,4]
    X_test = points[120:150,0:4]
    y_test = points[120:150,4]'''
#Alternatively we can use the iris dataset from the scikitlearn
if __name__ == "__main__":
    X=load_iris().data
    y=load_iris().target
    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=3)
    y_pred=np.zeros(len(y_test))
    #Based on the coefficients that we get from run() function predict the 
    #output for X_test
    c,m1,m2,m3,m4 = training(X_train,y_train)
    for i in range(0, len(X_test)):
         x1=X_test[i,0]
         x2=X_test[i,1]
         x3=X_test[i,2]
         x4=X_test[i,3]
         y_pred[i]=m1*x1+m2*x2+m3*x3+m4*x4+c
               
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    Z = y_pred
    #Variance is calculated using the formula  for more details please refer
    #https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics
    #/regression.py#L444
    r2_score = 0
    avg=0
    tavg=0
    t_score=0
    for i in range(0, len(X_test)):
        avg += y_test[i]-y_pred[i]
        tavg+=y_test[i]
    avg=avg/len(X_test)
    tavg=tavg/len(X_test)
    for i in range(0, len(X_test)):
        r2_score +=((y_test[i]-y_pred[i])-avg)**2
        t_score +=(y_test[i]-tavg)**2
    r2_score=1-r2_score/t_score
    print("Variance_score={0}".format(r2_score))
    plt.scatter(y_test, Z, c=y_test, cmap=cmap_bold, edgecolor='k')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title("Linear Regression (Predicted vs Actual)\n")
    plt.show()
