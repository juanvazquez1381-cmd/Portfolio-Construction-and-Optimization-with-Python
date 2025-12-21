import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# From Prices to Returns

prices_a = [8.70, 8.91, 8.71]

prices_a = np.array(prices_a)
print(prices_a)
print("--------------------------------------------------------")
print(prices_a[1:]/prices_a[:-1] - 1)
print("--------------------------------------------------------")

prices = pd.DataFrame({"BLUE": [8.70, 8.91, 8.71, 8.43, 8.73],
                       "ORANGE": [10.66, 11.08, 10.71, 11.59, 12.11]})

print(prices)
print("--------------------------------------------------------")
# because Pandas DataFrames will align the row index (in this case: 0, 1, 2, 3, 4)
# the exact same code fragment will not work as you might expect

print(prices.iloc[1:])
print("--------------------------------------------------------")

print(prices.loc[prices["BLUE"] < 8.91])
print("--------------------------------------------------------")

print(prices.loc[prices["ORANGE"] < 11.00])
print("--------------------------------------------------------")

print(prices.iloc[:-1])
print("--------------------------------------------------------")

# Wrong way to calculate returns
print(prices.iloc[1:]/prices.iloc[:-1] - 1)
print("--------------------------------------------------------")

# we can extract the values of the DataFrame column which returns a numpy array, so that the DataFrame
# does not try and align the rows.
print(prices.iloc[1:].values/prices.iloc[:-1] - 1)
print("--------------------------------------------------------")

# Alternative way
print(prices.iloc[1:]/prices.iloc[:-1].values - 1)
print("--------------------------------------------------------")


print(prices.shift(1))
print("--------------------------------------------------------")

returns = prices/prices.shift(1)-1
print(returns)

print("--------------------------------------------------------")
returns = returns.dropna()
print(returns)
print("--------------------------------------------------------")

prices = pd.read_csv('sample_prices.csv')
print(prices)
print("--------------------------------------------------------")

returns = prices.pct_change()
returns = returns.dropna()
print(returns)
print("--------------------------------------------------------")

print(returns.mean())
print("--------------------------------------------------------")

print(returns.std())
print("--------------------------------------------------------")

returns.plot.bar()
print("--------------------------------------------------------")

prices.plot()
print("--------------------------------------------------------")

# Compounded Returns
print(returns + 1)
print("--------------------------------------------------------")

print(np.prod(returns+1))
print("--------------------------------------------------------")

print((returns + 1).prod())
print("--------------------------------------------------------")

print((returns + 1).prod()- 1)
print("--------------------------------------------------------")

print((((returns+1).prod()-1)*100).round(2))
print("--------------------------------------------------------")

# Monthly Returns
rm = 0.01
print((1+rm)**12 - 1)
print("--------------------------------------------------------")

# Quaterly Returns
rq = 0.04
print((1+rq)**4 - 1)
print("--------------------------------------------------------")

# Daily Returnx
rd = 0.0001
print((1+rd)**252 - 1)
print("--------------------------------------------------------")

plt.show()
