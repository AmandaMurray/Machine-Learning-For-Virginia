#Hey look we have an ML Document where we can maybe probably understand WTF is happening :)

#Because we need all these ALL THE FUCKING TIME
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

#For data cleaning+Processing+Discovery
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer

#For Cross-Evaluation when tuning hyperparameter
from sklearn.model_selection import cross_val_score

#For SVMs
from sklearn.svm import LinearSVC #We probably won't have linear data
from sklearn.svm import SVC

#For evaluating a classfication algorithm
from sklearn.metrics import precision_score, recall_score, accuracy_score #These take .predict !!!Use np.ravel(y_test) rather than y_test
from sklearn.metrics import roc_curve #Remember to send the SVC.decision_function to this rather than the .predict


## Import, clean, prepare the data, look at the correlations
    ## This is the point we're we'd be creating the "Busy-ness" feature

database = pd.read_csv("database.csv")
print(database)
#Call the weird pandas thing that categorizes things.

pipe = Pipeline([ ('imputer', Imputer( strategy ="median")),
                         ('std_scaler', StandardScaler()),
                         (' selector', pd.DataFrameSelector( cat_attribs)),
                         (' cat_encoder', CategoricalEncoder( encoding =" onehot-dense")), ])
X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)


##Classification algorithm
    ##Try a linear SVC for a few C
for i in (1,10,100,1000):
    print(i)
#Try it in the possible ranges.

    ##Kernelize the SVC. We don't know how these work, but we'll try a few different ones and pick the one with the best performance

    ##Optimize whichever performed the best above using cross-evaluation


    ##Display and understand the data


##Logistics Regression Algorithm
    ##We'll have to do research for optimizing and changing forms on this, but it may just be adjusting the regularization hyperparameter


## Clustering Algorithm
    ## We know nothing about this yet. Hopefully we'll learn it before we have to do it


##Decision trees?
    ##I think decisions trees might be a valid approach to this problem?
