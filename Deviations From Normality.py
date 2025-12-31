import pandas as pd
import numpy as np
import scipy.stats
import edhec_risk_kit as erk

hfi = erk.get_hfi_returns()
print(hfi.head())
print("--------------------------------------------------------------------------------------")

print(pd.concat([hfi.mean(), hfi.median(), hfi.mean()>hfi.median()], axis = 'columns'))
print("--------------------------------------------------------------------------------------")

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or series
    """

    demeaned_r = r - r.mean()

    # Use the population standard deviation, so set dof = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**3).mean()
    return exp/(sigma_r**3)

print(skewness(hfi).sort_values())
print("--------------------------------------------------------------------------------------")

print("You get a similar answer as the one above but it will not be order")
print(scipy.stats.skew(hfi))
print("--------------------------------------------------------------------------------------")

print(hfi.shape)
print("--------------------------------------------------------------------------------------")

normal_rets = np.random.normal(0, 0.15, (263,1))
print("Normal mean:" , normal_rets.mean(), "Standard deviation:", normal_rets.std())
print("--------------------------------------------------------------------------------------")

print(erk.kurtosis(hfi))
print("--------------------------------------------------------------------------------------")

print(scipy.stats.kurtosis(hfi))
print("--------------------------------------------------------------------------------------")

# Note that these numbers are all lower by 3 from the number we have computed. That's because,
# as we said above, the expected kurtosis of a normally distributed series of numbers is 3,
# and scipy.stats is returning the Excess Kurtosis.

print(scipy.stats.kurtosis(normal_rets))
print("--------------------------------------------------------------------------------------")

print(erk.kurtosis(normal_rets))
print("--------------------------------------------------------------------------------------")

print(scipy.stats.jarque_bera(normal_rets))
print("--------------------------------------------------------------------------------------")

print(scipy.stats.jarque_bera(hfi))
print("--------------------------------------------------------------------------------------")

isinstance(hfi, pd.DataFrame)
print("--------------------------------------------------------------------------------------")

print(erk.is_normal(normal_rets))
print("--------------------------------------------------------------------------------------")

# Testing CRSP SmallCap and LargeCap returns

ffme = erk.get_ffme_returns()
print(erk.skewness(ffme))
print("--------------------------------------------------------------------------------------")

print(erk.kurtosis(ffme))
print("--------------------------------------------------------------------------------------")

print(erk.is_normal(ffme))
print("--------------------------------------------------------------------------------------")
