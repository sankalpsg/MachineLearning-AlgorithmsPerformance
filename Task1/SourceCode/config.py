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

newYorkTaxi_DecisionTreeClassifier=0
newYorkTaxi_LinearRegression=0
newYorkTaxi_LogisticRegression=0
newYorkTaxi_RandomForest=0

sumwithnoise_LinearRegression=0
sumwithnoise_DecisionTreeClassifier=0
sumwithnoise_LogisticRegression=0
sumwithnoise_RandomForest=0

Sum_withOUTnoise_LinearRegression=0
Sum_withOUTnoise_DecisionTreeClassifier=0
Sum_withOUTnoise_LogisticRegression=0
Sum_withOUTnoise_RandomForest=0

YearPrediction_LinearRegression=0
YearPrediction_DecisionTreeClassifier=0
YearPrediction_LogisticRegression=0
YearPrediction_RandomForest=1

use_anonymous = True