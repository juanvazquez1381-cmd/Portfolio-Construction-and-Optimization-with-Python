import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

data =  pd.read_csv('Portfolios_Formed_on_ME_monthly_EW.csv', header = 0,
                   index_col = 0, parse_dates = True, na_values = 99.99)

cols = ['Lo 20', 'Hi 20']
twenty_rtns = data[cols]
twenty_rtns = twenty_rtns/100
twenty_rtns.plot()
print("--------------------------------------------------------------")

# Questoins 1 to 4
months = twenty_rtns.shape[0]
low20_rtns = round((twenty_rtns['Lo 20'] + 1).prod()**(12/months) - 1, 5)*100
print(f"Annualized for Lo 20 returns is {low20_rtns}")
print("--------------------------------------------------------------")

low20_vox = round((twenty_rtns['Lo 20'].std()*np.sqrt(12) ),5) * 100
print(f"Annualized volatility for Lo 20 is {low20_vox}")
print("--------------------------------------------------------------")

hi20_rtns = round((twenty_rtns['Hi 20'] + 1).prod()**(12/months) - 1, 5)*100
print(f"Annualized returns for Hi 20 is {hi20_rtns}")
print("--------------------------------------------------------------")

hi20_vox = round((twenty_rtns['Hi 20'].std()*np.sqrt(12) ),5) * 100
print(f"Annualized volatility for Hi 20 is {hi20_vox}")
print("--------------------------------------------------------------")

# Questions 4 to 8
twenty_rtns.index = pd.to_datetime(twenty_rtns.index, format = "%Y%m")

slice_low20 = twenty_rtns.loc["1999":"2015", 'Lo 20' ]
months = slice_low20.shape[0]
slice_low20_rtns = round((slice_low20 + 1).prod()**(12/months)-1,5) * 100
print("Annual returns for Lo 20 for 1999-2015", slice_low20_rtns)
print("--------------------------------------------------------------")

slice_low20_vox = round(slice_low20.std()*np.sqrt(12),5)*100
print("Annualized Volatility for Low 20 for 1999-2015", slice_low20_vox)
print("--------------------------------------------------------------")

slice_hi20 = twenty_rtns.loc["1999":"2015", 'Hi 20' ]
months = slice_hi20.shape[0]
slice_hi20_rtns = round((slice_hi20 + 1).prod()**(12/months)-1,5) * 100
print("Annual returns for Hi 20 for 1999-2015", slice_hi20_rtns)
print("--------------------------------------------------------------")

slice_hi20_vox = round(slice_hi20.std()*np.sqrt(12),5)*100
print("Annualized Volatility for Hi 20 for 1999-2015", slice_hi20_vox)
print("--------------------------------------------------------------")

# Questions 9 to 12

def drawdown(return_series: pd.Series):
    """"Takes a time series of asset returns.
    Returns a DataFrame with columns for the
    wealth index, the previous peaks and the
    percentage drawdown"""

    wealth_index = 1000*(1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = round((wealth_index - previous_peaks) / previous_peaks,5)

    return pd.DataFrame({"Wealth": wealth_index,
                         "Previous Peak": previous_peaks,
                         "Drawdown": drawdowns})

low20_draw = drawdown(slice_low20).min()
hi20_draw = drawdown(slice_hi20).min()

print("Low 20 drawdown:")
print(low20_draw)
print("--------------------------------------------------------------")

print("High 20 drawdown:")
print(hi20_draw)
print("--------------------------------------------------------------")

print(drawdown(slice_low20)["Drawdown"].idxmin())
print(drawdown(slice_hi20)["Drawdown"].idxmin())
print("--------------------------------------------------------------")

# Questons 13 to 16
hfi = erk.get_hfi_returns()
hfi_slice = hfi.loc["2009":"2018"]
print(erk.semideviation(hfi_slice).sort_values())
print("--------------------------------------------------------------")

hfi_slice2 = hfi.loc["2009":]
print(erk.semideviation(hfi_slice2).sort_values())
print("--------------------------------------------------------------")


print(erk.skewness(hfi_slice).sort_values())
print("--------------------------------------------------------------")

hfi_slice3 = hfi.loc["2000":"2018"]
print(erk.kurtosis(hfi_slice3).sort_values())
print("--------------------------------------------------------------")

