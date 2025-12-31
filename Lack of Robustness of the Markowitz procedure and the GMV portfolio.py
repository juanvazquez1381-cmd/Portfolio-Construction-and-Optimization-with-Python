import edhec_risk_kit as erk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ind = erk.get_ind_returns()
er = erk.annualize_rets(ind["1996":"2000"], 12)
cov = ind["1996":"2000"].cov()

l = ["Food", "Steel"]
res1 = erk.msr(0.1, np.array(er[l]),cov.loc[l,l])
print(res1)
print("-----------------------------------------------------------")

res2 = np.round(res1,4)
print(res2)
print("-----------------------------------------------------------")

res3 = er[l]
print(res3)
print("-----------------------------------------------------------")

res4 = erk.msr(0.1, np.array([.11,.12]),cov.loc[l,l])
print(res4)
print("-----------------------------------------------------------")

res5 = erk.msr(0.1, np.array([.10, .13]), cov.loc[l,l])
print(res5)
print("-----------------------------------------------------------")

res6 = erk.msr(0.1, np.array([.13, .10]), cov.loc[l,l])
print(res6)
print("-----------------------------------------------------------")

erk.plot_ef(20, er, cov, show_cml=True, riskfree_rate = 0.1)
plt.show()

erk.plot_ef(20, er, cov, show_cml=True, riskfree_rate=0.1, show_ew=True)
plt.show()
