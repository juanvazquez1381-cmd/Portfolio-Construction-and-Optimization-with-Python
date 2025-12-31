import edhec_risk_kit as erk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ind = erk.get_ind_returns()
er = erk.annualize_rets(ind["1996":"2000"], 12)
cov = ind["1996":"2000"].cov()

# Plot EF
ax = erk.plot_ef(20, er, cov)
ax.set_xlim(left=0)
plt.show()

# Get MSR
ax = erk.plot_ef(20, er, cov)
ax.set_xlim(left=0)
rf = 0.1
w_msr = erk.msr(rf, er, cov)
r_msr = erk.portfolio_return(w_msr,er)
vol_msr = erk.portfolio_vol(w_msr,cov)

# Add CML
cml_x = [0, vol_msr]
cml_y = [rf, r_msr]
ax.plot(cml_x, cml_y, color = 'green', marker = 'o', linestyle = 'dashed',linewidth =2, markersize = 12)
plt.show()


erk.plot_ef(20, er, cov, style='-', show_cml=True, riskfree_rate=0.1)
plt.show()

