# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:38:49 2017

@author: Smarth Katyal
"""
def executeAlgo( datasetDirectory , codebaseDir ):
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	import os
	import datetime as dt
	newDir = datasetDirectory+'train.csv'
	print("Dataset Being Used:",newDir)
	# Loading the dataset
	dataset = pd.read_csv(newDir,parse_dates=[2,3])
	dataset['long_or_short'] = np.where(dataset['trip_duration']>=500, '1', '0')
	R = 6373.0
	#from math import sin, cos, sqrt, atan2, radians
	lat1 = np.radians(dataset.iloc[:, 6:7].values)
	lon1 = np.radians(dataset.iloc[:, 5:6].values)
	lat2 = np.radians(dataset.iloc[:, 8:9].values)
	lon2 = np.radians(dataset.iloc[:, 7:8].values)
	
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	
	a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
	
	dataset["distance"] = R * c
	
	
	#Function to convert date to unix timestamp
	def dt2ut(dt):
		epoch = pd.to_datetime('1970-01-01')
		return (dt - epoch).total_seconds()
	
	#Convert dates to unix timestamp
	dataset['pickup_datetime'] = dataset['pickup_datetime'].apply(dt2ut).astype(np.int64)
	dataset['dropoff_datetime'] = dataset['dropoff_datetime'].apply(dt2ut).astype(np.int64)
	
	#print("Result:", distance)
	
	#Diving the dataset as per question guidelines
	
	D1 = dataset.iloc[0:100,:].values
	D1=pd.DataFrame(D1)
	
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
	
	
	
	print(" Start : For D1===================================================\n\n")
	exit
	#creating a matrix of features and target feature
	X1 = D1.iloc[:, [2,3,12]].values
	y1 = D1.iloc[:, 11].values
	
	#Diving the dataset through 10 KFolds    
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score,precision_score
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D1===================================================\n\n")
	
	
	
	print(" Start : For D2===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D2.iloc[:, [2,3,12]].values
	y1 = D2.iloc[:, 11].values
	
	#Diving the dataset through 10 KFolds     
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D2===================================================\n\n")
	
	
	print(" Start : For D3===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D3.iloc[:, [2,3,12]].values
	y1 = D3.iloc[:, 11].values
	
	#Diving the dataset through 10 KFolds     
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D3===================================================\n\n")
	
	print(" Start : For D4===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D4.iloc[:, [2,3,12]].values
	y1 = D4.iloc[:, 11].values
	
	#Diving the dataset through 10 KFolds     
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D4===================================================\n\n")
	
	print(" Start : For D5===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D5.iloc[:, [2,3,12]].values
	y1 = D5.iloc[:, 11].values
	
	#Diving the dataset through 10 KFolds     
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D5===================================================\n\n")
	
	print(" Start : For D6===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D6.iloc[:, [2,3,12]].values
	y1 = D6.iloc[:, 11].values
		
	#Diving the dataset through 10 KFolds 	
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D6===================================================\n\n")
	
	
	print(" Start : For D7===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D7.iloc[:, [2,3,12]].values
	y1 = D7.iloc[:, 11].values
	
	#Diving the dataset through 10 KFolds     
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D7===================================================\n\n")
	
	
	
	print(" Start : For D8===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D8.iloc[:, [2,3,12]].values
	y1 = D8.iloc[:, 11].values
	
	#Diving the dataset through 10 KFolds     
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		#print("RMSE", mean_squared_error(y1_test,y1_pred))
		#print("R2 SCORE", r2_score(y1_test,y1_pred))
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D8===================================================\n\n")
	
	
	print(" Start : For D9===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D9.iloc[:, [2,3,12]].values
	y1 = D9.iloc[:, 11].values
	
	#Diving the dataset through 10 KFolds     
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	
	#Initializing the metric values
	
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
		
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		mean = mean + precision_score(y1_test, y1_pred,average='macro')   #Precision Score
		
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	print("Mean Precision Score",mean)
	print("Mean Accuracy", meanacc)
	print("Count",count)
	print("End: For D9===================================================\n\n")
	return 1;