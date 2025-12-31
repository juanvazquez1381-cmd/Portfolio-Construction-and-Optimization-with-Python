import pandas as pd
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

ind = pd.read_csv("ind30_m_vw_rets.csv", header = 0, index_col = 0)/100
ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')

print(ind.head())
print("-----------------------------------------------------------------------------")

print(ind.columns)
print("-----------------------------------------------------------------------------")

ind.columns = ind.columns.str.strip()
print(ind.columns)
print("-----------------------------------------------------------------------------")

ind = erk.get_ind_returns()
print(ind.shape)
print("-----------------------------------------------------------------------------")

erk.drawdown(ind["Food"])["Drawdown"].plot.line(figsize = (12, 6))
plt.show()

res1 = erk.var_gaussian(ind[["Food", "Beer", "Smoke"]], modified=True)
print(res1)
print("-----------------------------------------------------------------------------")

erk.var_gaussian(ind).sort_values().plot.bar(figsize = (12,6))
plt.show()

erk.sharpe_ratio(ind, 0.03, 12).sort_values().plot.bar(figsize =(12,6),title="Industry Sharpe Ratios 1926-2018")
plt.show()

erk.sharpe_ratio(ind["2000":], 0.03, 12).sort_values().plot.bar(figsize =(12,6),title='Industry Sharpe Ratios since 2000')
plt.show()

er = erk.annualize_rets(ind["1995":"2000"],12) # expected returns
print(er)
print("-----------------------------------------------------------------------------")
er.sort_values().plot.bar(figsize = (12,6))
plt.show()

cov = ind["1995":"2000"].cov()
print(cov.shape)
print("-----------------------------------------------------------------------------")
print(cov)
