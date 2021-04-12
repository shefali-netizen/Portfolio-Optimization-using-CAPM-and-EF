# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:47:58 2021

@author: shefa
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:11:51 2021

@author: shefa
"""

import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
from scipy import stats

df = data.DataReader(['AAPL', 'NKE', 'GOOGL', 'AMZN','FB','^GSPC'], 'yahoo', start='2015/01/01', end='2020/12/31')
df = df['Adj Close']

df_SP = pd.DataFrame(data={'SP':df['^GSPC']})
df.drop(['^GSPC'],axis=1,inplace=True)

print(df.head())
print(df_SP.head())

rf = 0.01
#beta calculation
b=dict()
rm = df_SP.mean()
ER = dict()
for stock in df.columns:
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_SP['SP'], df[stock])
    b[stock] = slope
    ER[stock] = rf + (b[stock] * (rm-rf))
    ER[stock] = ER[stock]['SP']


ER = pd.Series(ER)
print(ER)

cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
# Yearly returns for individual companies
ind_er = df.resample('Y').last().pct_change().mean()

print(type(ind_er))
print(ind_er)

#####
#w = [0.1, 0.1, 0.3, 0.2,0.2,0.1]




ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

#print(ann_sd)

assets = pd.concat([ER , ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']
#print(assets)

#####

p_ret = [] # Define an empty array for portfolio returns
p_volatility = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(df.columns)
num_portfolios = 10000


for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ER) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
                                    
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    p_volatility.append(ann_sd)
    #print(weights.shape)
    
data = {'Returns':p_ret, 'Volatility':p_volatility}

for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]
portfolios  = pd.DataFrame(data)
print(portfolios.head()) # Dataframe of the 10000 portfolios created

# Plot efficient frontier


min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]


rf = 0.01 # risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
#optimal_risky_port

plt.subplots(figsize=(10, 10))
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)

plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
  






