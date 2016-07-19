# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import math

df = pd.read_csv('/Users/Chang/Desktop/data_science/loansData.csv')

# Data Cleaning

df = df.dropna()
df.index = range(len(df))


g = lambda x: round(np.float(x.rstrip('%')),4)
for i in range(0,len(df)):
    df.set_value(i,'Interest.Rate',g(df['Interest.Rate'].loc[i]))
    
df['Interest.Rate']= pd.to_numeric(df['Interest.Rate'])
    
f = lambda x: round(float(x.rstrip('months')),4)
for i in range(0,len(df)):
    df.set_value(i,'Loan.Length',f(df.iloc[i,3]))  
    
df['Loan.Length']= pd.to_numeric(df['Loan.Length'])

df['FICO.Score']=pd.Series(np.random.randn(len(df)))

for i in range(0,len(df)):
    s = df.iloc[i,9]
    s = s.split('-')
    df.iloc[i,14]=float(s[0])
    
    
# Logistic Regression

df['IR_TF']=0
for i in range(len(df)):
    if df.loc[i,'Interest.Rate'] >= 12:
        df.set_value(i,'IR_TF',1)
df['intercept'] = 1.0

ind_vars = df[['Amount.Requested','FICO.Score','intercept']]
logit = sm.Logit(df['IR_TF'],ind_vars).fit()

#print(logit.summary())

# def logistic function

def logistic_function(x,y):
    return 1/(1+math.exp(-logit.params[0]*x - logit.params[1]*y - logit.params[2]))
    
print("The probability of get <12% interest rate is {}".format(str(1-logistic_function(10000,720))))
print("As P>0.7, we will get the loan.")

x=[]
y=[]
pp = pd.DataFrame(x,y,columns=[['x','y']])
pp.x = np.arange(550,800)
for i in range(len(pp)):
    pp.loc[i,'y'] = logistic_function(10000,pp.loc[i,'x'])

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
ax.plot(pp.x,1-pp.y)   
ax.set_ylabel('probability of getting interest.rate lower than 12%, yes = 1, no = 0')
ax.set_xlabel('FICO.SCore')