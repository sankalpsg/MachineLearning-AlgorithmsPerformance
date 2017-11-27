# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Thu Oct 26 20:40:01 2017

@author: Smarth Katyal
"""

import os 
import config as cfg
import zipfile
directory=cfg.parent_dataset_directory
print(directory);
codebaseDir = os.getcwd()
decider = cfg.newYorkTaxi_DecisionTreeClassifier
if(decider==1):
    print("\n\n*************Execution for Decision Tree Classifier on New York Taxi Dataset Started...\n Please wait for results****************\n\n");
    import NewYorkTaxi_DecisionTreeClassifier as nd
    datasetDirectory = directory+'New York City Taxi Trip Duration/';
    zipDirectory = datasetDirectory + 'New York City Taxi Trip Duration.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for Decision Tree Classifier on New York Taxi Dataset Completed...****************\n\n");
    
decider = cfg.newYorkTaxi_LinearRegression
if(decider==1):
    print("\n\n*************Execution for LinearRegression on New York Taxi Dataset Started...\n Please wait for results****************\n\n");
    import NewYorkTaxi_LinearRegression as nl
    datasetDirectory = directory+'New York City Taxi Trip Duration/';
    zipDirectory = datasetDirectory + 'New York City Taxi Trip Duration.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LinearRegression on New York Taxi Dataset Completed...****************\n\n");
    
decider = cfg.newYorkTaxi_LogisticRegression
if(decider==1):
    print("\n\n*************Execution for LogisticRegression on New York Taxi Dataset Started...\n Please wait for results****************\n\n");
    import NewYorkTaxi_LogisticRegression as nl
    datasetDirectory = directory+'New York City Taxi Trip Duration/';
    zipDirectory = datasetDirectory + 'New York City Taxi Trip Duration.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LogisticRegression on New York Taxi Dataset Completed...****************\n\n");
    

decider = cfg.newYorkTaxi_RandomForest
if(decider==1):
    print("\n\n*************Execution for RandomForestRegression on New York Taxi Dataset Started...\n Please wait for results****************\n\n");
    import NewYorkTaxi_RandomForestRegression as nl
    datasetDirectory = directory+'New York City Taxi Trip Duration/';
    zipDirectory = datasetDirectory + 'New York City Taxi Trip Duration.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for RandomForestRegression on New York Taxi Dataset Completed...****************\n\n");



decider = cfg.sumwithnoise_LinearRegression
if(decider==1):
    print("\n\n*************Execution for LinearRegression on The Sum(With Noise) Dataset Started...\n Please wait for results****************\n\n");
    import Sum_withnoise_LinearRegression as nl
    datasetDirectory = directory+'The SUM dataset/with noise/';
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LinearRegression on The Sum(With Noise) Dataset Completed...****************\n\n");


decider = cfg.sumwithnoise_DecisionTreeClassifier
if(decider==1):
    print("\n\n*************Execution for DecisionTreeClassifier on The Sum(With Noise) Dataset Started...\n Please wait for results****************\n\n");
    import Sum_withnoisewithoutkfold_DecisionTreeClassification as nl
    datasetDirectory = directory+'The SUM dataset/with noise/';
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for DecisionTreeClassifier on The Sum(With Noise) Dataset Completed...****************\n\n");



decider = cfg.sumwithnoise_LogisticRegression
if(decider==1):
    print("\n\n*************Execution for LogisticRegression on The Sum(With Noise) Dataset Started...\n Please wait for results****************\n\n");
    import Sum_withnoisewithoutkfold_LogisticRegression as nl
    datasetDirectory = directory+'The SUM dataset/with noise/';
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LogisticRegression on The Sum(With Noise) Dataset Completed...****************\n\n");



decider = cfg.sumwithnoise_RandomForest
if(decider==1):
    print("\n\n*************Execution for RandomForest on The Sum(With Noise) Dataset Started...\n Please wait for results****************\n\n");
    import Sum_withnoise_RandomForest as nl
    datasetDirectory = directory+'The SUM dataset/with noise/';
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for RandomForest on The Sum(With Noise) Dataset Completed...****************\n\n");



decider = cfg.Sum_withOUTnoise_LinearRegression
if(decider==1):
    print("\n\n*************Execution for LinearRegression on The Sum(Without Noise) Dataset Started...\n Please wait for results****************\n\n");
    import Sum_withOUTnoise_LinearRegression as nl
    datasetDirectory = directory+'The SUM dataset/without noise/';
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LinearRegression on The Sum(Without Noise) Dataset Completed...****************\n\n");



decider = cfg.Sum_withOUTnoise_DecisionTreeClassifier
if(decider==1):
    print("\n\n*************Execution for DecisionTreeClassification on The Sum(Without Noise) Dataset Started...\n Please wait for results****************\n\n");
    import Sum_withOUTnoise_withoutKFold_DecisionTreeClassification as nl
    datasetDirectory = directory+'The SUM dataset/without noise/';
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for DecisionTreeClassification on The Sum(Without Noise) Dataset Completed...****************\n\n");



decider = cfg.Sum_withOUTnoise_LogisticRegression
if(decider==1):
    print("\n\n*************Execution for LogisticRegression on The Sum(Without Noise) Dataset Started...\n Please wait for results****************\n\n");
    import Sum_withOUTnoisewithoutkfold_LogisticRegression as nl
    datasetDirectory = directory+'The SUM dataset/without noise/';
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LogisticRegression on The Sum(Without Noise) Dataset Completed...****************\n\n");



decider = cfg.Sum_withOUTnoise_RandomForest
if(decider==1):
    print("\n\n*************Execution for RandomForestRegression on The Sum(Without Noise) Dataset Started...\n Please wait for results****************\n\n");
    import Sum_withOUTnoise_RandomForest as nl
    datasetDirectory = directory+'The SUM dataset/without noise/';
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for RandomForestRegression on The Sum(Without Noise) Dataset Completed...****************\n\n");


#########################

decider = cfg.YearPrediction_LinearRegression
if(decider==1):
    print("\n\n*************Execution for LinearRegression on MillionSong Year Dataset Started...\n Please wait for results****************\n\n");
    import YearPrediction_LinearRegression as nl
    datasetDirectory = directory+'MillionSong Year-Prediction Dataset (Excerpt)/';
    zipDirectory = datasetDirectory + 'YearPredictionMSD.txt.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LinearRegression on MillionSong Year Dataset Completed...****************\n\n");
        


decider = cfg.YearPrediction_DecisionTreeClassifier
if(decider==1):
    print("\n\n*************Execution for DecisionTreeClassifier on MillionSong Year Dataset Started...\n Please wait for results****************\n\n");
    import YearPrediction_DecisionTreeClassifier as nl
    datasetDirectory = directory+'MillionSong Year-Prediction Dataset (Excerpt)/';
    zipDirectory = datasetDirectory + 'YearPredictionMSD.txt.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for DecisionTreeClassifier on MillionSong Year Dataset Completed...****************\n\n");
        
        

decider = cfg.YearPrediction_LogisticRegression
if(decider==1):
    print("\n\n*************Execution for LogisticRegression on MillionSong Year Dataset Started...\n Please wait for results****************\n\n");
    import YearPrediction_LogisticRegression as nl
    datasetDirectory = directory+'MillionSong Year-Prediction Dataset (Excerpt)/';
    zipDirectory = datasetDirectory + 'YearPredictionMSD.txt.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LogisticRegression on MillionSong Year Dataset Completed...****************\n\n");
        
    
decider = cfg.YearPrediction_RandomForest
if(decider==1):
    print("\n\n*************Execution for RandomForestRegression on MillionSong Year Dataset Started...\n Please wait for results****************\n\n");
    import YearPrediction_RandomForest as nl
    datasetDirectory = directory+'MillionSong Year-Prediction Dataset (Excerpt)/';
    zipDirectory = datasetDirectory + 'YearPredictionMSD.txt.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nl.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for RandomForestRegression on MillionSong Year Dataset Completed...****************\n\n");
    