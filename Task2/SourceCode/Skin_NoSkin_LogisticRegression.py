# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:20:12 2017

@author: Sankalp
"""
def executeAlgo( datasetDirectory , codebaseDir ):
	#To get the timestamp of machine
	import time
	ts = time.time()
	print (ts)
	
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	import os
	
	
	newDir = datasetDirectory+'Skin_NonSkin.txt'
	print("Dataset Being Used:",newDir)
	dataset = pd.read_csv(newDir,header = -1, delim_whitespace=True)
	
	dataset[3] = np.where(dataset[3]==1, 0, 1)
	
	#creating a matrix of features and target feature
	X1 = dataset.iloc[:, :-1].values
	y1 = dataset.iloc[:, 3].values
	
	from sklearn.cross_validation import train_test_split
	X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
	
	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X1_train = sc.fit_transform(X1_train)
	X1_test = sc.transform(X1_test)
	
	# Fitting Logistic Regression to the Training set
	from sklearn.linear_model import LogisticRegression
	classifier = LogisticRegression(random_state = 0)
	classifier.fit(X1_train, y1_train)
	
	# Predicting the Test set results
	y1_pred = classifier.predict(X1_test)
	
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y1_test, y1_pred)
	
	
	from sklearn.model_selection import cross_val_score
	Accuracy = cross_val_score(estimator= classifier,X= X1_train, y = y1_train, cv=10, scoring= "accuracy")
	Precision = cross_val_score(estimator= classifier,X= X1_train, y = y1_train, cv=10, scoring= "precision")
	ROC = cross_val_score(estimator= classifier,X= X1_train, y = y1_train, cv=10, scoring= "roc_auc")
	Recall = cross_val_score(estimator= classifier,X= X1_train, y = y1_train, cv=10, scoring= "recall")
	F1 = cross_val_score(estimator= classifier,X= X1_train, y = y1_train, cv=10, scoring= "f1")
	
	Accuracy = Accuracy.mean()
	Precision = Precision.mean()
	ROC = ROC.mean()
	Recall = Recall.mean()
	F1 = F1.mean()
	
	print("Accuracy : ", Accuracy)
	print("Precision : ", Precision)
	print("ROC : ", ROC)
	print("Recall : ", Recall)
	print("F1 : ", F1)
	
	
	from sklearn.metrics import make_scorer
	
	#defined Number_CorrectPred metric which finds the number of false predictions
	def my_custom_pred_func(y1_test, y1_pred):
			diff = np.abs(y1_test - y1_pred)
			diff = diff.sum()
			#print(diff)
			a = np.log(1+ diff)
			#print(a)
			return a
	
	
	Number_CorrectPred  = make_scorer(my_custom_pred_func)
	Number_CorrectPred = Number_CorrectPred(estimator= classifier,X= X1_train,y_true= y1_train)
	print("No of Correct Predictions_log: ",Number_CorrectPred)
	
	#made timediff metric which finds the run time of algorithm
	newts = time.time()
	timediff = newts - ts
	print ("RunTime: ", timediff)
	return 1