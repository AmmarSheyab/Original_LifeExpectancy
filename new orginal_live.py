# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:44:02 2022

@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Original_LifeExpectancy.csv')

df.info()
'''
i have in data 
2:object
4:int
16:float
'''
a=df.isnull().sum()


df.iloc[:,:].fillna(0,inplace=True)
#How many times is the value repeated?
b=df.nunique()

#mix values
from pandas.api.types import infer_dtype
mix=df.apply(lambda x: 'mixed' in infer_dtype(x))

df= pd.get_dummies(df,columns=['Country'],drop_first=True)
df= pd.get_dummies(df,columns=['Year'],drop_first=True)



X = df.loc[:,df.columns.difference(['Life expectancy '],sort=False)].values
y = df.iloc[:,1].values

'''
d1 = death.iloc[:,:3].values
d2 = death.iloc[:,4:].values

df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)

X = pd.concat([df1, df2], axis=1)
y=death.iloc[:,3].values

'''








from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])

'''
df1= pd.get_dummies(X[:,0],columns=['Country'],drop_first=True)
df2= pd.get_dummies(X[:,1],columns=['Year'],drop_first=True)

X=np.append(X,df1,axis=1)
X=np.append(X,df2,axis=1)
X=X[:,3:]

'''

'''
from sklearn.impute import SimpleImputer
simpleImputer=SimpleImputer(missing_values=0,strategy='mean')
simpleImputer.fit(X[:,1:20])
(X[:,1:20])=simpleImputer.transform((X[:,1:20]))
'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=(0))

from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()
X_train=standardScaler.fit_transform(X_train)
X_test=standardScaler.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train,y_train)
y_pred=linearRegression.predict(X_test)

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 


X=np.append(np.ones((len(X),1)).astype(int),values=X,axis=1)#CONSTANT

import statsmodels.api as sm



def reg_ols(X,y):
    columns=list(range(X.shape[1]))
    
    for i in range(X.shape[1]):
        X_opt=np.array(X[:,columns],dtype=float) 
        regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
        pvalues = list(regressor_ols.pvalues)
        d=max(pvalues)
        if (d>0.05):
            for k in range(len(pvalues)):
                if(pvalues[k] == d):
                    del(columns[k])  
    
    return(X_opt,regressor_ols)

X_opt,regressor_ols=reg_ols(X, y)
regressor_ols.summary()



from sklearn.model_selection import train_test_split
X_train_opt,X_test_opt,y_train_opt,y_test_opt=train_test_split(X_opt,y,test_size=0.2,random_state=(0))
'''
from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()
X_train_opt=standardScaler.fit_transform(X_train_opt)
X_test_opt=standardScaler.fit_transform(X_test_opt)
'''
from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train_opt,y_train_opt)
y_pred1=linearRegression.predict(X_test_opt)

print('#'*50)

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test_opt, y_pred1), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test_opt, y_pred1), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test_opt, y_pred1), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test_opt, y_pred1), 2)) 


from sklearn.model_selection import train_test_split
X_log_train, X_log_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_log_train, y_train)

# Predicting the Test set results
y_log_pred = classifier.predict(X_log_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_log_pred)
print(cm)




