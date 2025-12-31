import edhec_risk_kit as erk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ind = erk.get_ind_returns()
er = erk.annualize_rets(ind["1996":"2000"], 12)
cov = ind["1996":"2000"].cov() # THis the covirance matrix

l = ["Food", "Beer", "Smoke", "Coal"]
print(er[l])
print("------------------------------------------------")
print(cov.loc[l,l]) # Create Cov Matrix
print("------------------------------------------------")
ew = np.repeat(0.25, 4)

print(erk.portfolio_return(ew, er[l]))
print("------------------------------------------------")

print(erk.portfolio_vol(ew, cov.loc[l,l]))
print("------------------------------------------------")

n_points = 20
weights = [np.array([w, 1-w]) for w in np.linspace(0,1,n_points)]
type(weights)
len(weights)
weights[0]
weights[4]
weights[19]

l = ["Games", "Fin"]
rets = [erk.portfolio_return(w, er[l]) for w in weights]
vols = [erk.portfolio_vol(w, cov.loc[l,l]) for w in weights]
ef = pd.DataFrame({"R":rets, "V": vols})
ef.plot.scatter(x="V", y = "R")
plt.show()

l = ["Fin", "Beer"]
cov2_asset = cov.loc[l,l]
erk.plot_ef2(25, er[l].values, cov2_asset)
plt.show()
