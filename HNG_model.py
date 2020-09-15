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

"""=========================== User input variables ==========================="""
time = 200 #s  - input desired duration of simulation

# Initial conditions
n_0 = 10  #input how many nucleii are already present
c_k = 1.2 #concentrations of solute - should be in kmol/m³
T = 298 #K
V_elect = 0.0005  #volume electrolyte m³

# Thermo properties
c_k_sat = 1 # Solute concentration at 100% saturation - should be in kmol/m³
surf_energy = 0.54 #J / m² surface energy of solid product, should be Li₂O₂, temporary data for lithium, ... (http://crystalium.materialsvirtuallab.org/) can probably be done with cantera?
MW = 45.881 #kg/kmol
den =2310 #kg/m³ 2.31 #g/cm³

# Growth reaction parameters
k_grow = 2    # Rate coefficient (mol/m²/s)
n = 1           # Reaction order (-)
k_r = 1   # Rate coefficient

# Constants
R = 8314.4 #J/K / kmol
"""=================== Initial calculations, problem set-up ==================="""

# Initial parameter calculations:
S = c_k/c_k_sat
mol_vol = den/MW #kmol/m³

#intializing the solution vector
initial = {}
initial['r_0'] = 2**surf_energy*(mol_vol)/(R*T*m.log(S)) #m
#initial['n_0'] =0 # nucleii
initial['S'] = S #unitless
sol_vec = list(initial.values())  # solution vector
print(sol_vec)
#Thermo Adjustments
k_rev = k_r*m.exp(2*surf_energy*mol_vol/(R*T*initial['r_0']))

A_spec = (10*initial['r_0'])**2/(V_elect)
#Reaction surface area/volume of electrolyte, used if the rate of reactions is mol/m², I think used in nucleation, Specific surface of reaction (m²/m³) using r_0 for scale

int_volume =  2/3*initial['r_0']**3
#initiual volume of a nucleation

#%%

"""======================== Define the residual function ========================"""
def residual(t, solution):
    r, s = solution #indicates variable array because I forget
    dr_dt = MW/den*(k_grow*(s)**n-k_rev*(2*m.pi*r**2))
    ds_dt = - n_0*(dr_dt * 2 * m.pi * r**2)*mol_vol/V_elect/c_k_sat # distribute concentration change into total electrolyte
#    drad_dt = (.5*m.tanh(180*(conc-1)+.5))*mol_vol*k_grow*(conc-1)**n
#    dconc_dt = - (.5*m.tanh(180*(conc-1)+.5))*n_0*(drad_dt*2*m.pi*radius**2)/mol_vol/V_elect/co_k
#    drad_dt = m.tanh(30*(conc-1))*mol_vol*k_grow*(conc-1)**n
#    dconc_dt = - m.tanh(30*(conc-1))*n_0*(drad_dt*2*m.pi*radius**2)/mol_vol/V_elect/co_k  distrute change in mass to electrolyte
    return [dr_dt, ds_dt]

"""========================== Run the simulation =========================="""
solution = solve_ivp(residual, [0, time], sol_vec) #growth senario

"""============================ Post-processing ============================"""


radius = solution.y[0]
concentrations = solution.y[1]
print(concentrations)
t = solution.t

#%%

r_range = max(radius) + min(radius)
max_rad = max(radius)
x = [ran.random()*r_range for i in range(n_0)]
y = [ran.random()*r_range for i in range(n_0)]

#%%
with PdfPages('output' +  dt.datetime.now().strftime("%Y%m%d") + '.pdf') as pdf:
    plt.figure(0)
    plt.plot(t,radius)
    plt.xlabel("Time (s)")
    plt.ylabel("Radius (m)")
    pdf.savefig()
    plt.show()
    plt.close()

    plt.figure(1)
    plt.plot(t,concentrations)
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (C/Ck)")
    pdf.savefig()
    plt.show()
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
