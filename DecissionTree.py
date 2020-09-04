# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:10:21 2020

@author: Adam
"""

#type of job categorical
#marital : marital status categorical
#education categorical
#housing: has housing loan?
#loan: has personal loan?

import pandas as pd 
df = pd.read_csv('C:/Users/Adam/Desktop/Projekty/DataScience Library/Decission Tree/bankloans.csv',sep= ';') 
df = df.dropna()

X = df.iloc[:,[0,1,2,3,5,6]]
Y = pd.DataFrame(df.iloc[:, 20])
df.iloc[:, 20].value_counts()

Y['y'] = Y['y'].apply(lambda x: 1 if x == "yes" else 0)
X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100, stratify=Y)

from sklearn.tree import DecisionTreeClassifier 
clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=4, min_samples_leaf=10)
clf_gini.fit(X_train, y_train) 
y_pred_gini = clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred_gini))     
print ("Accuracy : ", accuracy_score(y_test,y_pred_gini)*100)      
print("Report : ", classification_report(y_test, y_pred_gini))

import matplotlib.pyplot as plt
from sklearn import tree
fig, ax = plt.subplots(figsize=(12, 12))
feature_cols = list(X.columns) 
tree.plot_tree(clf_gini,max_depth=6, fontsize=10, feature_names = feature_cols,filled=True, class_names=["No agreement","Agreement"]) 
plt.show()

y_pred_gini = pd.DataFrame(y_pred_gini, columns = ["Agreement"])
result = y_test.join(y_pred_gini, how='inner')