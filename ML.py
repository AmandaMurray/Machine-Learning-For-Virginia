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

#This is where we read in the original Database
#database = pd.read_csv("databse.csv")
#Creating feature that says how many murders were reported in the agency that year
'''
database['murders_that_year'] = 0
this_agency = ""
count = 1
start_row = 0
end_row = 0

for index, row in database.iterrows():
    if (this_agency != row['Agency Name']):
        this_agency = row['Agency Name']

        #iterate from start row to end row and set murders_that_month to count
        for i in range(start_row-1, end_row+1):
            database.iloc[i, database.columns.get_loc('murders_that_year')] = count

        count = 1
        start_row = end_row + 1
        end_row = end_row +1

    else:
        count = count +1
        end_row = end_row +1

#Save the outputted database to a CSV.
database.to_csv('out.csv')
print("finished")
'''

#Read in our newly created Database
database = pd.read_csv('out.csv')
#print(database)
database = database.drop(labels=['Agency Name', 'Crime Type'], axis = 1)

#Drop some data to prevent memory errors(RIP our wimpy laptops)
use, throw_away = train_test_split(database, test_size = 0.75, random_state = 42)
database = use
#print(database.shape) #(159613, 10)
#Fine at this level. Cool

non_cat_labels = ['Victim Age', 'murders_that_year', 'Year']
database_only_cat = database.drop(labels = non_cat_labels, axis=1)
#Okay here.
database_numeric = database[non_cat_labels]
#Looks good here.

pipe = Pipeline([('onehot', OneHotEncoder(sparse = False)),])
database_only_cat = pipe.fit_transform(database_only_cat) #<<< I cant make this line work
#print(database_only_cat)
temp = pd.DataFrame(database_only_cat)
#print(temp.shape)#(159613, 48)
#print(temp)
print(database_numeric)
print(database_numeric.shape)
database = pd.concat([temp,database_numeric], axis=1, join="outer", ignore_index = True)
#i dont think we should ignore the index
#can we try it with the inner join???
#Size = (40176, 51)(BAD)dear fucking god, yeah try that
#Size = ((279050, 51)) STILL BAD, i guess try both?
# Size = (279050, 51) Still some bad bad-sauce
#do we have an id column? Numeric has indexes. But no ID Column
print(database.shape)#(279050, 51)
database.to_csv('aahhhh.csv') #I'm pushing this CSV.
#print(database)

'''
#print(database_only_cat)
#Basic train-set splitting
train_set, test_set = train_test_split(database, test_size = 0.2, random_state = 42) #Also we need a way to split this by Virginia vs Not Virginia.
X_train = train_set.drop(columns=["Crime Solved"])
y_train = train_set[['Crime Solved']]
X_test = test_set.drop(columns=["Crime Solved",])
y_test = test_set[['Crime Solved']]

# pipe =

#Our pipeline.

##Classification algorithm
    ##Try a linear SVC for a few C
maxScore = 0;
maxVal = 0;
for i in (0, 1,10,100,1000):
    svm_clf = LinearSVC(C=i, loss="hinge", random_state=42, tol=10, max_iter=5000)
    scores = cross_val_score(svm_clf, X_train, np.ravel(y_train), cv=5, scoring='f1')
    mean = scores.mean()
    print(mean)
    if(mean > maxScore):
        maxScore = mean
        maxVal = i

print("The best C value is: " + str(maxVal) + " which gives us a score of: " + str(maxScore))
'''
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
