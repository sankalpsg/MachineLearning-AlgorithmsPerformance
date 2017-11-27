# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 02:01:18 2017

@author: Smarth Katyal
"""
def executeAlgo( datasetDirectory , codebaseDir ):
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	import os
	
	#Reading dataset to memory
	newDir = datasetDirectory+'The SUM dataset, with noise.csv'
	print("Dataset Being Used:",newDir)
	dataset = pd.read_csv(newDir)
	#os.getcwd()
	#os.chdir("D:\College\Machine Learning\Assignment\Assignment1\DataSet")
	#filename = 'The SUM dataset, with noise.csv'
	#dataset = pd.read_csv(filename)
	
	#Encoding dependent variable
	y_encode = dataset.iloc[:, 12].values
	from sklearn.model_selection import KFold
	from sklearn.metrics import accuracy_score,precision_score,average_precision_score
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.preprocessing import StandardScaler
	from math import sqrt
	from sklearn.metrics import log_loss
	from sklearn.preprocessing import LabelEncoder
	labelencoder_y = LabelEncoder()
	dataset.iloc[:,12] = labelencoder_y.fit_transform(y_encode)
	
	#Dividing the dataset
	D1 = dataset.iloc[0:100,:].values
	D1 = pd.DataFrame(D1)
	
	D2 = dataset.iloc[0:500,:].values
	D2=pd.DataFrame(D2)
	
	D3 = dataset.iloc[0:1000,:].values
	D3=pd.DataFrame(D3)
	
	D4 = dataset.iloc[0:5000,:].values
	D4=pd.DataFrame(D4)
	
	D5 = dataset.iloc[0:10000,:].values
	D5=pd.DataFrame(D5)
	
	D6 = dataset.iloc[0:50000,:].values
	D6=pd.DataFrame(D6)
	
	D7 = dataset.iloc[0:100000,:].values
	D7=pd.DataFrame(D7)
	
	D8 = dataset.iloc[0:500000,:].values
	D8=pd.DataFrame(D8)
	
	D9 = dataset.iloc[0:1000000,:].values
	D9=pd.DataFrame(D9)
	
	
	
	ss = StandardScaler()
	regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	
	##############################################
	########   For D1 ##########################
	##############################################
	
	print("Start :: For D1=======================================")
	#creating a matrix of features and target feature
	X1 = D1.iloc[:, 1:11].values
	y1 = D1.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	
	
	##############################################
	########   For D2 ##########################
	##############################################
	
	print("Start :: For D2=======================================")
	#creating a matrix of features and target feature
	X1 = D2.iloc[:, 1:11].values
	y1 = D2.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	##############################################
	########   For D3 ##########################
	##############################################
	
	print("Start :: For D3=======================================")
	#creating a matrix of features and target feature
	X1 = D3.iloc[:, 1:11].values
	y1 = D3.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	##############################################
	########   For D4 ##########################
	##############################################
	
	print("Start :: For D4=======================================")
	#creating a matrix of features and target feature
	X1 = D4.iloc[:, 1:11].values
	y1 = D4.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	##############################################
	########   For D5 ##########################
	##############################################
	
	print("Start :: For D5=======================================")
	#creating a matrix of features and target feature
	X1 = D5.iloc[:, 1:11].values
	y1 = D5.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	##############################################
	########   For D6 ##########################
	##############################################
	
	print("Start :: For D6=======================================")
	#creating a matrix of features and target feature
	X1 = D6.iloc[:, 1:11].values
	y1 = D6.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	##############################################
	########   For D7 ##########################
	##############################################
	
	print("Start :: For D7=======================================")
	#creating a matrix of features and target feature
	X1 = D7.iloc[:, 1:11].values
	y1 = D7.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	##############################################
	########   For D8 ##########################
	##############################################
	
	print("Start :: For D8=======================================")
	#creating a matrix of features and target feature
	X1 = D8.iloc[:, 1:11].values
	y1 = D8.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	##############################################
	########   For D9 ##########################
	##############################################
	
	print("Start :: For D9=======================================")
	#creating a matrix of features and target feature
	X1 = D9.iloc[:, 1:11].values
	y1 = D9.iloc[:, 12].values
	
	#Start:: Commment below code for K Fold cross validation
	from sklearn.cross_validation import train_test_split
	X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3,random_state = 0)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	#Scaling the features
	X1_train = ss.fit_transform(X1_train)
	X1_test = ss.transform(X1_test)
	
	regressor.fit(X1_train, y1_train)
	y1_pred = regressor.predict(X1_test)
	y1_test = y1_test.astype(int)
	y1_pred = y1_pred.astype(int)
	
	print("Accuracy score", accuracy_score(y1_test, y1_pred))
	print("Precision Score", precision_score(y1_test, y1_pred,average='macro'))
	return 1