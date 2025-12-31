import edhec_risk_kit as erk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ind = erk.get_ind_returns()
er = erk.annualize_rets(ind["1996":"2000"], 12)
cov = ind["1996":"2000"].cov()

l = ["Games", "Fin"]
assets = cov.loc[l,l]
erk.plot_ef2(20, er[l].values, assets)
plt.show()

weights_15 = erk.minimize_vol(0.15, er[l], cov.loc[l,l])
vol_15 = erk.portfolio_vol(weights_15, cov.loc[l,l])
print(vol_15)

a = ["Smoke", "Fin", "Games", "Coal"]
assets2 = cov.loc[a,a]
erk.plot_ef(50, er[a], cov.loc[a,a])
plt.show()
