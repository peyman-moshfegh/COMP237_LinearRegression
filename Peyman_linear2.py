# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:57:05 2021

@author: Peyman
"""

'''part a.i'''
import pandas as pd
import os

filename = "Ecom Expense.csv"
path = "C:\\Users\\Peyman\\.spyder-py3\\assignment3"
fullpath = os.path.join(path, filename)
ecom_exp_peyman = pd.read_csv(fullpath)

'''part b.i'''
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(ecom_exp_peyman.head(3))

'''part b.ii'''
print(ecom_exp_peyman.shape)

'''part b.iii'''
for i in ecom_exp_peyman.columns:
    print(i)

'''part b.iv'''
print(ecom_exp_peyman.dtypes)

'''part b.v'''
import numpy as np
missing = [0 for i in range(ecom_exp_peyman.shape[1])]
for i in range(ecom_exp_peyman.shape[0]):
    if ecom_exp_peyman.loc[i].isnull().any():
        for j in range(ecom_exp_peyman.shape[1]):
            if ecom_exp_peyman.loc[i].isnull()[j]:
                missing[j] += 1
j = 0
for i in ecom_exp_peyman.columns:
    print("{0: <20}{1}".format(i, missing[j]))
    #print(i, missing[j])
    j += 1

'''part c.iv'''
ecom_exp_peyman = ecom_exp_peyman.drop(columns=["Transaction ID"])

'''parts c.i-ii-iii'''
ecom_exp_peyman = pd.get_dummies(ecom_exp_peyman)

'''part c.v'''
def Normalize(df):
    cols = df.columns
    min = df.min()
    max = df.max()
    df = df.astype(float)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            df.at[i, cols[j]] = (df.loc[i][j] - min[j]) / (max[j] - min[j])
    return df

'''part c.vi'''
ecom_exp_peyman = Normalize(ecom_exp_peyman)

'''part c.vii'''
print(ecom_exp_peyman.head(2))

'''part c.viii'''
ecom_exp_peyman.hist(figsize = (9, 10))

'''part c.ix'''
cols = ['Age','Monthly Income','Transaction Time','Total Spend']
pd.plotting.scatter_matrix(ecom_exp_peyman[cols], alpha=0.4, figsize=(13, 15))

'''parts d.i-ii-iii'''
from sklearn.model_selection import train_test_split
train, test = train_test_split(ecom_exp_peyman , test_size = 0.35, random_state = 8)

'''part d.iv'''
x_train_peyman = train.drop(columns=['Age', 'Items', 'Record', 'Total Spend'])
y_train_peyman = train.drop(columns=[x for x in ecom_exp_peyman if x != 'Total Spend'])
x_test_peyman = test.drop(columns=['Age', 'Items', 'Record', 'Total Spend'])
y_test_peyman = test.drop(columns=[x for x in ecom_exp_peyman if x != 'Total Spend'])

'''part d.v'''
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train_peyman, y_train_peyman)

'''part d.vi'''
print(reg.coef_)

'''part d.vii'''
print(reg.score(x_train_peyman, y_train_peyman))

'''parts d.viii-ix-x-xi'''
x_train_peyman = train.drop(columns=['Age', 'Items', 'Total Spend'])
y_train_peyman = train.drop(columns=[x for x in ecom_exp_peyman if x != 'Total Spend'])
x_test_peyman = test.drop(columns=['Age', 'Items', 'Total Spend'])
y_test_peyman = test.drop(columns=[x for x in ecom_exp_peyman if x != 'Total Spend'])

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train_peyman, y_train_peyman)

print(reg.coef_)

print(reg.score(x_train_peyman, y_train_peyman))