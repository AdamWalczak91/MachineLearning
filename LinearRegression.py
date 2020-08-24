# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 00:50:22 2020

@author: Adam Walczak
"""

import pandas as pd  
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import levene
from scipy.stats import ks_2samp
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt  

def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)


df = pd.DataFrame(pd.read_csv('C:/Users/Adam/Desktop/Projekty/DataScience Library/Regression/Linear regression/Admission_Predict_Ver1.1.csv', delimiter=','))

encoder = OneHotEncoder()

encoded_df = encoder.fit_transform(df[['Research']]).toarray()
df['Research - no'] = encoded_df[:,0]
df['Research - yes'] = encoded_df[:,1]
df = df.drop(columns = ['Research'])

df['Z score - GRE'] = stats.zscore(df['GRE Score'])
df['Z score - TOEFL'] = stats.zscore(df['TOEFL Score'])
df.shape

df = df.loc[df['Z score - GRE'].abs()<=3]
df = df.loc[df['Z score - TOEFL'].abs()<=3]
df.shape

scaling = StandardScaler()
df['GRE Score'] = scaling.fit_transform(df[['GRE Score']])
df['TOEFL Score'] = scaling.fit_transform(df[['TOEFL Score']])
df['University Rating'] = scaling.fit_transform(df[['University Rating']])

X = df[['GRE Score','TOEFL Score','University Rating', 'Research - no', 'Research - yes']]
Y = df[['Chance of Admit ']]

calc_vif(X[['GRE Score','TOEFL Score','University Rating']])
X = X.drop(columns = ['TOEFL Score'])

x1 = X.iloc[:,0].values
x2 = X.iloc[:,1].values

stat, p = levene(x1,x2) 
print('%.8f' % p)

stat, p = ks_2samp(x1, x2)
print('%.8f' % p)

stat, p = f_oneway(x1, x2)
print('%.8f' % p)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

SGDReg = linear_model.SGDRegressor(max_iter = 1000, tol = 1e-3)
SGDReg.fit(X_train, Y_train)
Y_pred = pd.DataFrame(SGDReg.predict(X_test)).reset_index(drop=True)
Y_test.reset_index(drop=True, inplace = True)

print('R2 score: ',r2_score(Y_test, Y_pred))
print('Intercept: ',SGDReg.intercept_)
print('Coef: ',SGDReg.coef_)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

results = pd.DataFrame({'Actual': Y_test.iloc[:, 0].head(25), 'Predicted': Y_pred.iloc[:, 0].head(25)},)
results.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

Prediction_variables = np.array(X.iloc[0,:].values).reshape(-1, 1).transpose()
print(SGDReg.predict(Prediction_variables))