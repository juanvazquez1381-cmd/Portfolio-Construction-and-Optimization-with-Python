import pandas as pd
import edhec_risk_kit as erk
hfi = erk.get_hfi_returns()
print(hfi.head())
print("--------------------------------------------------------------------------------------")
print(hfi.tail())

