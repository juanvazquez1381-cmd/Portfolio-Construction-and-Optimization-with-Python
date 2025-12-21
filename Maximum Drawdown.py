import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

me_m = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",
                   header = 0, index_col = 0,
                   parse_dates=True, na_values = 99.99)

rets = me_m[['Lo 10', 'Hi 10']]
rets.columns = ["SmallCap", "LargeCap"]
rets = rets/100
rets.plot.line()

print(rets.head())
print("-----------------------------------------------------------------------")

print(rets.index)
print("-----------------------------------------------------------------------")

rets.index = pd.to_datetime(rets.index, format = "%Y%m")
print(rets.index)
print("-----------------------------------------------------------------------")

print("The returns for 2008")
print(rets.loc["2008"]) 
print("-----------------------------------------------------------------------")

rets.index = rets.index.to_period('M')
print(rets.head())
print("-----------------------------------------------------------------------")

print(rets.info())
print("-----------------------------------------------------------------------")

print(rets.describe())
print("-----------------------------------------------------------------------")

wealth_index = 1000*(1 + rets["LargeCap"]).cumprod()
wealth_index.plot()
print("-----------------------------------------------------------------------")

previous_peaks = wealth_index.cummax()
previous_peaks.plot()
plt.show()
print("-----------------------------------------------------------------------")

drawdown = (wealth_index - previous_peaks)/ previous_peaks
drawdown.plot()
plt.show()
print("----------------------------------------------------------------------")

print(drawdown.min())
print("----------------------------------------------------------------------")

drawdown.loc["1975":].plot()
plt.show()
print("----------------------------------------------------------------------")

print(drawdown['1975':].min())
print("----------------------------------------------------------------------")

def drawdown(return_series: pd.Series):
    """"Takes a time series of asset returns.
    Returns a DataFrame with columns for the
    wealth index, the previous peaks and the
    percentage drawdown"""

    wealth_index = 1000*(1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame({"Wealth": wealth_index,
                         "Previous Peak": previous_peaks,
                         "Drawdown": drawdowns})

print(drawdown(rets["LargeCap"]).head())
print("----------------------------------------------------------------------")


print(drawdown(rets["LargeCap"]).min())
print("----------------------------------------------------------------------")

print(drawdown(rets["SmallCap"]).min())
print("----------------------------------------------------------------------")

print(drawdown(rets["LargeCap"])["Drawdown"].idxmin())
print("----------------------------------------------------------------------")

print(drawdown(rets["SmallCap"])["Drawdown"].idxmin())
print("----------------------------------------------------------------------")

print(drawdown(rets["LargeCap"]["1975":])["Drawdown"].idxmin())
print("----------------------------------------------------------------------")

print(drawdown(rets["SmallCap"]["1975":])["Drawdown"].idxmin())
print("----------------------------------------------------------------------")

print(drawdown(rets["SmallCap"]["1975":])["Drawdown"].min())

