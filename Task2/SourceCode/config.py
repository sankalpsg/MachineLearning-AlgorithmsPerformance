# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
This is the configuration file for the project
-- Please provide the parent directory where the datasets are present as per format on Dropbox.
-- The wrapper script will handle the task of unzipping the data, so do not unzip the data
-- The file gives flexibility to execute, any number of algorithms at one time. Processing will be done sequentially and not parallely.
-- To execute an algo for a particular dataset, set its property value to '1'.
-- To skip execution of a particular algo on a particular dataset, set its property to '0'.
-- You can set multiple properties as 1. 

-- Some algorithms take quite a long time, so suggestion is to execute only one algo at a time.
-- The result metrics will be visible in console 
"""
parent_dataset_directory="D:/College/Machine Learning/Assignment/DataSetsDownloaded/"

Share_Predict_A3_Logistic=1
SharePredict_A3_DecisionClassifier=1
SharePredict_A3_RandomForestClassifier=1
SharePredict_A3_KNearestNeighbour=1

Skin_NoSkin_LogisticRegression=1
Skin_NoSkin_DecisionClassifier=1
Skin_NoSkin_RandomForestClassifier=1
Skin_NoSkin_KNearestNeighbour=1



use_anonymous = True