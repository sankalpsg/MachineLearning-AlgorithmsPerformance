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
decider = cfg.Share_Predict_A3_Logistic
if(decider==1):
    print("\n\n*************Execution for LogisticRegression on OnlineNewsPopularity dataset...\n Please wait for results****************\n\n");
    import Share_Predict_A3_Logistic as nd
    datasetDirectory = directory+'OnlineNewsPopularity/';
    zipDirectory = datasetDirectory + 'OnlineNewsPopularity.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LogisticRegression on OnlineNewsPopularity Dataset Completed...****************\n\n");
    

decider = cfg.SharePredict_A3_DecisionClassifier
if(decider==1):
    print("\n\n*************Execution for DecisionTreeClassifier on OnlineNewsPopularity dataset...\n Please wait for results****************\n\n");
    import SharePredict_A3_Decision as nd
    datasetDirectory = directory+'OnlineNewsPopularity/';
    zipDirectory = datasetDirectory + 'OnlineNewsPopularity.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for DecisionTreeClassifier on OnlineNewsPopularity Dataset Completed...****************\n\n");
    
    
    
decider = cfg.SharePredict_A3_RandomForestClassifier
if(decider==1):
    print("\n\n*************Execution for  on RandomForestClassifier dataset...\n Please wait for results****************\n\n");
    import SharePredict_A3_RandomForest as nd
    datasetDirectory = directory+'OnlineNewsPopularity/';
    zipDirectory = datasetDirectory + 'OnlineNewsPopularity.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for RandomForestClassifier on OnlineNewsPopularity Dataset Completed...****************\n\n");
    
    
    

decider = cfg.SharePredict_A3_KNearestNeighbour
if(decider==1):
    print("\n\n*************Execution for  on KNearestNeighbourClassification dataset...\n Please wait for results****************\n\n");
    import SharePredict_A3_Knn as nd
    datasetDirectory = directory+'OnlineNewsPopularity/';
    zipDirectory = datasetDirectory + 'OnlineNewsPopularity.zip'
    zip_ref = zipfile.ZipFile(zipDirectory, 'r')
    zip_ref.extractall(datasetDirectory)
    zip_ref.close()
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for KNearestNeighbourClassification on OnlineNewsPopularity Dataset Completed...****************\n\n");
    
    

decider = cfg.Skin_NoSkin_LogisticRegression
if(decider==1):
    print("\n\n*************Execution for LogisticRegression on SkinNonSkin dataset...\n Please wait for results****************\n\n");
    import Skin_NoSkin_LogisticRegression as nd
    datasetDirectory = directory+'SkinNonSkin/';
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for LogisticRegression on SkinNonSkin Dataset Completed...****************\n\n");
    
    
    

decider = cfg.Skin_NoSkin_DecisionClassifier
if(decider==1):
    print("\n\n*************Execution for DecisionTreeClassifier on SkinNonSkin dataset...\n Please wait for results****************\n\n");
    import Skin_NoSkin_DecisionTreeClassifier as nd
    datasetDirectory = directory+'SkinNonSkin/';
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for DecisionTreeClassifier on SkinNonSkin Dataset Completed...****************\n\n");


decider = cfg.Skin_NoSkin_RandomForestClassifier
if(decider==1):
    print("\n\n*************Execution for RandomForestClassifier on SkinNonSkin dataset...\n Please wait for results****************\n\n");
    import Skin_NoSkin_DecisionTreeClassifier as nd
    datasetDirectory = directory+'SkinNonSkin/';
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for RandomForestClassifier on SkinNonSkin Dataset Completed...****************\n\n");



decider = cfg.Skin_NoSkin_KNearestNeighbour
if(decider==1):
    print("\n\n*************Execution for KNearestNeighbourClassification on SkinNonSkin dataset...\n Please wait for results****************\n\n");
    import Skin_NoSkin_KNN as nd
    datasetDirectory = directory+'SkinNonSkin/';
    status = nd.executeAlgo(datasetDirectory,codebaseDir)
    if(status==1):
        print("\n\n*************Execution for KNearestNeighbourClassification on SkinNonSkin Dataset Completed...****************\n\n");

    