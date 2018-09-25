#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 22:40:01 2018

@author: varun
"""
#importing libraries
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense


#import the dataset
df = pd.read_csv("Churn_Modelling.csv")

print(df.describe())
print(df.head())

X = df.iloc[:,3:13].values
y = df.iloc[:,-1].values

#encode categorical variables
lb_geo = LabelEncoder()
X[:,1] = lb_geo.fit_transform(X[:,1])

lb_gender = LabelEncoder()
X[:,2] = lb_gender.fit_transform(X[:,2])

#creating dummy variables
ohe_geo = OneHotEncoder(categorical_features=[1])
X = ohe_geo.fit_transform(X).toarray()
X = X[:, 1:]

#split the dataset into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#perform feature scalling as there will be a lot of calculations involved and this will also
#prevent one independent variable dominating other variable
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#initialize an ANN
clf = Sequential()

#add input layer and first hidden layer
clf.add(Dense(units = 6, kernel_initializer= 'uniform', activation = 'relu', input_dim = 11))

#add second hidden layer
clf.add(Dense(units = 6, kernel_initializer= 'uniform', activation = 'relu'))

#add output variable
clf.add(Dense(units = 1, kernel_initializer= 'uniform', activation='sigmoid'))

#compile our ANN by applying stochastic gradient descent 
clf.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

#fit ANN to training set
clf.fit(x= X_train, y= y_train, batch_size= 10, epochs= 200)

#predict the test set results
y_pred = clf.predict(X_test)
y_pred = (y_pred >0.5)

#make confusion matrix
cm = confusion_matrix(y_test, y_pred)







