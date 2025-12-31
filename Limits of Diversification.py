import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolios Average number of firms
    """
    ind = pd.read_csv("ind30_m_nfirms.csv", header = 0, index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()

    return ind

def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolios Average Size (market cap)
    """

    ind = pd.read_csv("ind30_m_size.csv", header = 0, index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

ind_return = erk.get_ind_returns()
ind_nfirms = get_ind_nfirms()
ind_size = get_ind_size()

print(ind_nfirms.head())
print("-------------------------------------------------------------------")

print(ind_size.head())
print("-------------------------------------------------------------------")

print(ind_return.shape)
print("-------------------------------------------------------------------")

print(ind_size.shape)
print("-------------------------------------------------------------------")

ind_mktcap = ind_nfirms * ind_size

print(ind_mktcap.shape)
print("-------------------------------------------------------------------")

print(ind_mktcap.head())
print("-------------------------------------------------------------------")

total_mktcap = ind_mktcap.sum(axis=1)
total_mktcap.plot()
plt.show()

ind_capweight = ind_mktcap.divide(total_mktcap, axis = "rows")
res1 = all(abs(ind_capweight.sum(axis = "columns")-1) < 1E-10)
print(res1)
print("-------------------------------------------------------------------")


sects = ["Steel", "Fin"]
ind_capweight[sects].plot()
plt.show()

total_market_return = (ind_capweight * ind_return).sum(axis = "columns")
total_market_index = erk.drawdown(total_market_return).Wealth
total_market_index.plot(title = "Total Market Cap Weighted Index 1926-2018")
plt.show()

total_market_index["1980":].plot(figsize=(12,6))
total_market_index["1980":].rolling(window = 36).mean().plot()
plt.show()

# Let's create a time series of the annualized returns over the trailing 36 months
# and the average correlation across stocks over that same 36 months.

tmi_tr36rets = total_market_return.rolling(window=36).aggregate(erk.annualize_rets, periods_per_year=12)
tmi_tr36rets.plot(figsize = (12,5), label = "Trailing 36 mo Returns", legend = True)
total_market_return.plot(label = "Returns", legend = True)
plt.show()

# Let's start by contructing the time series of correlations over time over a 36
# month window.
ts_corr = ind_return.rolling(window = 36).corr()
res2 = ts_corr.tail()
print(res2)
print("-------------------------------------------------------------------")

ts_corr.index.names = ['date','industry']
print(ts_corr.tail())
print("-------------------------------------------------------------------")

ind_tr36corr = ts_corr.groupby(level = 'date').apply(lambda cormat: cormat.values.mean())

tmi_tr36rets.plot(secondary_y = True, legend = True, label="Tr 36 mo return", figsize=(12,6))
ind_tr36corr.plot(legend = True, label = "Tr 36 mo Avg Correlation")
plt.show()

print(tmi_tr36rets.corr(ind_tr36corr))

# Clearly, these two series are negatively correlated, which explains why diversification fails
# you when you need it most. When markets fall, correlations rise, making diversification
# much less valuable.


