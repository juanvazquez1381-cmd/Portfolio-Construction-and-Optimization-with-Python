#!/usr/bin/env python
# coding: utf-8

# # The Efficient Frontier - Part II
# 
# Let's start by loading the returns and generating the expected returns vector and the covariance matrix

# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import edhec_risk_kit_108 as erk

ind = erk.get_ind_returns()
er = erk.annualize_rets(ind["1996":"2000"], 12)
cov = ind["1996":"2000"].cov()


# As a first exercise, let's assume we have some weights, and let's try and compute the returns and volatility of a portfolio, given a set of weights, returns, and a covariance matrix.
# 
# The returns are easy, so let's add this to our toolkit
# 
# ```python
# 
# def portfolio_return(weights, returns):
#     """
#     Computes the return on a portfolio from constituent returns and weights
#     weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
#     """
#     return weights.T @ returns
# 
# ```
# 
# The volatility is just as easy in matrix form:
# 
# ```python
# def portfolio_vol(weights, covmat):
#     """
#     Computes the vol of a portfolio from a covariance matrix and constituent weights
#     weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
#     """
#     return (weights.T @ covmat @ weights)**0.5
# ```
# 
# 

# In[5]:


l = ["Food", "Beer", "Smoke", "Coal"]


# In[6]:


er[l]


# In[7]:


cov.loc[l,l]


# In[10]:


import pandas as pd
import numpy as np
ew = np.repeat(0.25, 4)
erk.portfolio_return(ew, er[l])


# In[11]:


erk.portfolio_vol(ew, cov.loc[l,l])


# # The 2-Asset Case
# 
# In the case of 2 assets, the problem is somewhat simplified, since the weight of the second asset is 1-the weight of the first asset.
# 
# Let's write a function that draws the efficient frontier for a simple 2 asset case.
# 
# Start by generating a sequence of weights in a list of tuples. Python makes it easy to generate a list by using something called a _list comprehension_ ... which you can think of as an efficient way to generate a list of values instead of writing a for loop.
# 

# In[12]:


import numpy as np

n_points = 20
weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]


# In[13]:


type(weights)


# In[14]:


len(weights)


# In[15]:


weights[0]


# In[16]:


weights[4]


# In[17]:


weights[19]


# In[18]:


l = ["Games", "Fin"]
rets = [erk.portfolio_return(w, er[l]) for w in weights]
vols = [erk.portfolio_vol(w, cov.loc[l,l]) for w in weights]
ef = pd.DataFrame({"R": rets, "V": vols})
ef.plot.scatter(x="V", y="R")


# We can create function that plots the frontier:
# 
# ```python
# def plot_ef2(n_points, er, cov):
#     """
#     Plots the 2-asset efficient frontier
#     """
#     if er.shape[0] != 2 or er.shape[0] != 2:
#         raise ValueError("plot_ef2 can only plot 2-asset frontiers")
#     weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
#     rets = [portfolio_return(w, er) for w in weights]
#     vols = [portfolio_vol(w, cov) for w in weights]
#     ef = pd.DataFrame({
#         "Returns": rets, 
#         "Volatility": vols
#     })
#     return ef.plot.line(x="Volatility", y="Returns", style=".-")
# ```
# 
# A useful summary of the visualization features in pandas is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html)
# 

# In[19]:


l = ["Fin", "Beer"]
erk.plot_ef2(25, er[l].values, cov.loc[l,l])


# In[ ]:




