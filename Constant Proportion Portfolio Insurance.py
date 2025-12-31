import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

ind_return = erk.get_ind_returns()
tmi_return = erk.get_total_market_index_returns()

risky_r = ind_return["2000":][["Steel", "Fin", "Beer"]]
# Assume the safe asset is paying 3% per year
safe_r = pd.DataFrame().reindex_like(risky_r)
safe_r.values[:] = 0.03/12 # fast way to set all values to a number
start = 1000 # start at $1000
floor = 0.80 # set the floor to 80 percent of the starting value

def compound1(r):
    return np.expm1(np.log1p(r).sum())

def compound2(r):
    return (r+1).prod()-1

sectors = ["Steel", "Fin", "Beer"]
print(compound1(ind_return[sectors]))
print("--------------------------------------------------------")

print(compound2(ind_return[sectors]))
print("--------------------------------------------------------")

# set up the CPPI parameters
dates = risky_r.index
n_steps = len(dates)
account_value = start
floor_value = start * floor
m = 3

# set up some DataFrames for saving intermediate values
account_history = pd.DataFrame().reindex_like(risky_r)
risky_w_history = pd.DataFrame().reindex_like(risky_r)
cushion_history = pd.DataFrame().reindex_like(risky_r)

for step in range(n_steps):
    cushion = (account_value - floor_value) / account_value
    risky_w = m * cushion
    risky_w = np.minimum(risky_w, 1)
    risky_w = np.maximum(risky_w, 0)
    safe_w = 1 - risky_w
    risky_alloc = account_value * risky_w
    safe_alloc = account_value * safe_w

    # recompute the new account value at the end of this step
    account_value = risky_alloc*(1 + risky_r.iloc[step]) + safe_alloc *(1 + safe_r.iloc[step])

    # save the histories for analyisis and plotting
    cushion_history.iloc[step] = cushion
    risky_w_history.iloc[step] = risky_w
    account_history.iloc[step] = account_value
    risky_wealth = start*(1 + risky_r).cumprod()

print(cushion_history.head())
print("--------------------------------------------------------")

print(risky_w_history.head())
print("--------------------------------------------------------")

print(account_history.head())
print("--------------------------------------------------------")

print(risky_wealth.head())
print("--------------------------------------------------------")
