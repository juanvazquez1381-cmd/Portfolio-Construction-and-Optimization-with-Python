import pandas as pd
import numpy as np
from scipy.stats import norm
import edhec_risk_kit as erk
import matplotlib.pyplot as plt

hfi = erk.get_hfi_returns()
print(erk.semideviation(hfi))
print("--------------------------------------------------------------------------")

print(hfi[hfi < 0].std(ddof = 0))
print("--------------------------------------------------------------------------")

print(erk.semideviation(hfi).sort_values())
print("--------------------------------------------------------------------------")

ffme = erk.get_ffme_returns()
print(erk.semideviation(ffme))
print("--------------------------------------------------------------------------")

print("Note that for reporting purposes, it is common to invert the sign so we report a positive number to represent the loss i.e. the amount that is at risk.")
print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
print(erk.var_historic(hfi, level = 1))
print("-------------------------------------------------------------------------------------------------------------------------------------------------------")

print(erk.cvar_historic(hfi, level = 1).sort_values())
print("--------------------------------------------------------------------------")

print(erk.cvar_historic(ffme))
print("--------------------------------------------------------------------------")

print(norm.ppf(.5))
print("--------------------------------------------------------------------------")

print(norm.ppf(.16))
print("--------------------------------------------------------------------------")

print(erk.var_gaussian(hfi))
print("--------------------------------------------------------------------------")

print(erk.var_historic(hfi))
print("--------------------------------------------------------------------------")


var_table = [erk.var_gaussian(hfi), 
             erk.var_gaussian(hfi, modified=True), 
             erk.var_historic(hfi)]
comparison = pd.concat(var_table, axis=1)
comparison.columns=['Gaussian', 'Cornish-Fisher', 'Historic']
comparison.plot.bar(title="Hedge Fund Indices: VaR at 5%")
plt.show()
print("--------------------------------------------------------------------------")

print(erk.skewness(hfi).sort_values(ascending=False))


