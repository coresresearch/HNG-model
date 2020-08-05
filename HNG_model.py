#%%
# Useful libraries
import datetime as dt
import shutil
import math as m
import numpy as np
import pandas as pd
import random as ran
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp

"""=========================== User imput variables ==========================="""
time = 200 #s  - input desired duration of simulation

# Initial conditions
n_0 = 0  #input how many nucleii are already present
c_k = 1.2 #concentratiosn of solute - should be in kmol/m³
T = 298 #K

# Solution, solute, and precipitate properties
c_k_sat = 1 # Solute concentration at 100% saturation - should be in kmol/m³
#TODO #1
surf_energy = 0.54 #J / m² surface energy of solid product, should be Li₂O₂, temporary data for lithium, ... (http://crystalium.materialsvirtuallab.org/) can probably be done with cantera?
MW = 45.881 #kg/kmol
den =2310 #kg/m³ 2.31 #g/cm³

# Growth reaction parameters
k_grow = 0.5    # Rate coefficient (mol/m²/s)
n = 2           # Reaction order (-)

# Simulation domain:
V_elect = 0.0005  #volume electrolyte m³

"""=================== Initial calculations, problem set-up ==================="""
# Constants
R = 8314.4 #J/K / kmol

# Initial parameter calculations:
S = c_k/c_k_sat
mol_vol = MW/den #m³/kmol

#intializing the solution vector
initial = {}
initial['r_0'] = 2**surf_energy*(1/mol_vol)/(R*T*m.log(S)) #m
#initial['n_0'] =0 # nucleii
initial['S'] = c_k/c_k_sat #unitless
sol_vec = list(initial.values())  # solution vector
print(sol_vec)

#TODO #2
A_spec = (10*initial['r_0'])**2/(V_elect)
#Reaction surface area/volume of electrolyte, used if the rate of reactions is mol/m², I think used in nucleation, Specific surface of reaction (m²/m³) using r_0 for scale

#TODO #3
int_volume =  2/3*initial['r_0']**3
#initiual volume of a nucleation

nucleii = 10
print (A_spec)

#%%

"""======================== Define the residual function ========================"""
def residual(t, solution):
    r, c_k = solution #indicates variable array because I forget
    dr_dt = mol_vol*k_grow*(c_k-1.)**n
    dc_dt = - nucleii*(dr_dt * 2 * m.pi * r**2)/mol_vol/V_elect/c_k_sat # distribute concentration change into total electrolyte
#    drad_dt = (.5*m.tanh(180*(conc-1)+.5))*mol_vol*k_grow*(conc-1)**n
#    dconc_dt = - (.5*m.tanh(180*(conc-1)+.5))*nuclii*(drad_dt*2*m.pi*radius**2)/mol_vol/V_elect/co_k
#    drad_dt = m.tanh(30*(conc-1))*mol_vol*k_grow*(conc-1)**n
#    dconc_dt = - m.tanh(30*(conc-1))*nuclii*(drad_dt*2*m.pi*radius**2)/mol_vol/V_elect/co_k  distrute change in mass to electrolyte
    return [dr_dt, dc_dt]

"""========================== Run the simulation =========================="""
solution = solve_ivp(residual, [0, time], sol_vec) #growth senario

"""============================ Post-processing ============================"""


radius = solution.y[0]
concentrations = solution.y[1]
print(concentrations)
t = solution.t

fig = plt.figure(0)
plt.plot(t,radius)
plt.figure(1)
plt.plot(t,concentrations)


#%%

r_range = max(radius) + min(radius)
max_rad = max(radius)
x = [ran.random()*r_range for i in range(nucleii)]
y = [ran.random()*r_range for i in range(nucleii)]

#%%
with PdfPages('output' +  dt.datetime.now().strftime("%Y%m%d") + '.pdf') as pdf:
    plt.figure(0)
    plt.plot(t,radius)
    pdf.savefig()
    plt.close()

    plt.figure(1)
    plt.plot(t,concentrations)
    pdf.savefig()
    plt.close()

    for i in range(0,len(t), int(0.01*max(t))):
        plt.figure(i+2)
        plt.scatter(x, y, s=np.ones_like(x)*3000*radius[i])
        plt.axis([0.0, max_rad, 0.0, max_rad])
        pdf.savefig()
        plt.close()
# %%

shutil.copy(__file__, __file__+ dt.datetime.now().strftime("%Y%m%d")+".txt")

# %%
