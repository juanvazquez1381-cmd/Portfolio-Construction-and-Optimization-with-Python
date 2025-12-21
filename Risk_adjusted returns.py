import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prices = pd.read_csv('sample_prices.csv')
returns = prices.pct_change()
returns = returns.dropna()
print(returns)
print("--------------------------------")

deviations = returns - returns.mean()
squared_deviations = deviations**2
variance = squared_deviations.mean()
volatility = np.sqrt(variance) # Population
print("Notice the outputs of volatility and reutrns.std() do not match")
print(volatility)
print("--------------------------------")

print(returns.std()) # Sample
print("--------------------------------")

# Long way to calculate volatility using sample standard deviation
number_of_obs = returns.shape[0]
deviations = returns - returns.mean()
squared_deviations = deviations**2
mean_squared_deviations = squared_deviations.sum()/(number_of_obs - 1)
volatility = np.sqrt(mean_squared_deviations)
print(volatility)
print("--------------------------------")

print(returns.std())
print("--------------------------------")

annualized_vol = returns.std()*(12**0.5)
print(annualized_vol)
print("--------------------------------")

# Risk Adjusted Returns for small caps and large caps

me_m = pd.read_csv('Portfolios_Formed_on_ME_monthly_EW.csv', header = 0,
                   index_col = 0, parse_dates = True, na_values = 99.99)

print(me_m.head())
print("--------------------------------")

cols = ['Lo 10', 'Hi 10']
returns = me_m[cols]
print(returns.head())
print("--------------------------------")


# Note that the data is already given in percentages (i.e 4.5 instead of 0.045)
# and we typically want to use the actual numbers (i.e. 0.045 instead of 4.5)

returns = returns/100
returns.columns = ['SmallCap', 'LargeCap']
returns.plot()
plt.show()
print("--------------------------------")

annualized_vol = returns.std()*np.sqrt(12)
print(annualized_vol)
print("--------------------------------")

# We can now compute the annualized returns as follows:
n_months = returns.shape[0]
return_per_month = (returns + 1).prod()**(1/n_months) - 1
print(return_per_month)
print("--------------------------------")

annualized_return = (returns + 1).prod()**(12/n_months) -1
print(annualized_return)
print("--------------------------------")

print(annualized_return/annualized_vol)
print("--------------------------------")

riskfree_rate = 0.03
excess_return = annualized_return - riskfree_rate
sharpe_ratio = excess_return/annualized_vol
print(sharpe_ratio)



