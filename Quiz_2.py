import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

# Questions 1 to 3
hfi = erk.get_hfi_returns()
data = hfi.loc['2000':]
print(data.head())
print("------------------------------------------------------------------")

Q1 = (erk.var_gaussian(data['Distressed Securities'],level = 1)*100).round(2)
print("Gaussian:", Q1)
print("------------------------------------------------------------------")

Q2 = (erk.var_gaussian(data['Distressed Securities'],level = 1, modified = True)*100).round(2)
print('Cornish-Fisher Adjustment:', Q2)
print("------------------------------------------------------------------")

Q3 = (erk.var_historic(data['Distressed Securities'], level = 1)*100).round(2)
print('Historic:', Q3)
print("----------------------------------------------------------------")

# Questions 4 to 10
data2 = erk.get_ind_returns()
inds = data2["2013":"2017"]
a = ["Books", "Steel", "Oil", "Mines"]
inds = inds[a]
cov = inds.cov() # Covaraince Matrix
sectors = cov.loc[a,a]
er = erk.annualize_rets(inds,12) # Annualize Returns
risk_free = .10

N = inds.shape[1]
equal_weights = round((1/N)*100,2)
Q4 = equal_weights
print('Weight of Steel:', Q4)
print("--------------------------------------------------------------")

placeh = erk.msr(risk_free, er, sectors) # MSR weights
Q5 = round(placeh.max()*100,2)
print('weight of the largest component of the MSR portfolio:', Q5)
print("--------------------------------------------------------------")

Q6 = a[placeh.argmax(axis=0)]
print('Largest componet of MSR Portfolio:',Q6)
print("-------------------------------------------------------------")

tolerance = 1e-6
Q7 = np.sum(np.abs(placeh) > tolerance)
print("Number of nonzero weights are:", Q7)
print("-------------------------------------------------------------")

pla = erk.gmv(sectors)# GMV weights
Q8 = round(pla.max()*100,2)
print('Weight of largest Componet of GMV:',Q8)
print("-------------------------------------------------------------")

Q9 = a[pla.argmax(axis=0)]
print("Largest componet of GMV are:", Q9)
print("--------------------------------------------------------------")

tolerance = 1e-6
Q10 = np.sum(np.abs(pla) > tolerance)
print("Number of nonzero weights are:", Q10)
print("--------------------------------------------------------------")

# Questions 11 to 12. We have to use the same weights as we did from previous
# questions

port_2018 = data2.loc["2018"][a]
cov_2018 = port_2018.cov()

placeh = erk.msr(risk_free, er, sectors) # MSR weights
pla = erk.gmv(sectors)# GMV weights

Q11 = (erk.portfolio_vol(placeh, cov_2018) * 12**0.5 *100).round(2)
print('The annualized volatility over 2018 for the MSR portfolio:', Q11)
print("-----------------------------------------------------------------")


Q12 = (erk.portfolio_vol(pla, cov_2018) * np.sqrt(12)*100).round(2)
print('The annualized volatility over 2018 for the GMV portfolio:', Q12)
print("-----------------------------------------------------------------")





