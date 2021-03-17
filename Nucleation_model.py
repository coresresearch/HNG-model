#%%
import datetime as dt
import shutil
import math as m
import numpy as np
import pandas as pd
import random as ran
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
#%%
"""=========================== Constants ==========================="""

k_B = 1.380649E-23 #J K-1 // Boltzmann's constant
ep_0 = 8.854E-12  #F m-1 // electric permittivity of a vacuum
N_a = 6.022E23 #mol-1 // Avogadro's number
R = 8.3145 #J mol-1 K-1 // Ideal constant

"""=========================== User system variables ==========================="""
#Parameters that can be changed during the trials
T = 25 + 273.15 #K // Temperature
C_LiO2_0 = 0.15 #mol m-3 // Electrolyte Concentration of LiO2 particles
C_Li_0 = 0.15 #mol m-3 // Electrolyte Concentration Li+ particles
Elyte_v = 500 #uL3 // Electrolyte Volume
N_0 = 0 # // Nucleations
A_0 = m.pi*0.009**2 #m2 // Cross sectional area
time = 100000 #s // simulation time

"""=========================== User thermodyanic imputs ==========================="""
#parameters that depend on the system being studied

C_LiO2_sat = 0.1 #mol m-3 // saturated concentration of LiO2 //Yin (2017)
C_Li_sat = 0.1 #mol m-3 // saturated concentration of LiO2
k_nu = 1E-6 #mol s-1 m-2 // kinetic rate constant for the nucleation reaction // Yin (2017)
gamma_surf = 7.7E-3 #J m-2 //  surface energy of the newly formed crystal phase// Danner (2019) - should be replaced for LiS system
V = 1.98E-5 #m3 mol-1 // molar volume// Yin (2017)
theta = 30*m.pi/180 # radians // Contact angle // Danner (2019) - should be replaced for LiS system
D_LiO2 = 1.2E-9 #m2 -s-1 // diffusion ceofficient // Yin (2017)


"""=========================== Thermodyanic system cals ==========================="""
Elyte_v_SI = Elyte_v*1.0E-9 #
phi = (2+m.cos(theta))*(1-m.cos(theta))**2/4  # - // contact angle correction factor

"""=========================== Initialization step ==========================="""
#%%
#intializing the solution vector
initial = {}
initial['N'] = N_0
initial['A'] = A_0
initial['C_Li'] = C_Li_0
initial['C_LiO2'] = C_LiO2_0
sol_vec = list(initial.values())  # solution vector
print(sol_vec)

"""=========================== Equations ==========================="""
#%%


def residual(t, solution):
    N, A, C_Li, C_LiO2 = solution
    a_d = (C_LiO2*N_a)**(-1/3) # length scale of diffusion
    r_crit = 2*gamma_surf*V/(R*T*m.log(C_LiO2/C_LiO2_sat*C_Li/C_Li_sat)) # m // critical radius
    N_crit = 4/3*m.pi*r_crit**3*N_a/V # number of molecules in the critical nucleus of size
    Del_G_Crit = phi*4/3*m.pi*gamma_surf*r_crit**2 # J mol-1 // energy barrier of the nucleation
    Z = m.sqrt(Del_G_Crit/(phi*3*m.pi*k_B*T*N_crit)) # - // Zeldovich factor
    V_crit = 4/3*m.pi*r_crit**3 # m3 // Critical volume
    N_sites = A/(m.pi*r_crit**2) # number of nucleation sites
    DN_Dt = D_LiO2*a_d*N_sites*Z*m.exp(-Del_G_Crit/(k_B*T))
    DA_Dt = - DN_Dt*m.pi*r_crit**2
    DCLi_Dt = -2*DN_Dt*V_crit/(V*Elyte_v_SI)
    DCLiO2_Dt = -2*DN_Dt*V_crit/(V*Elyte_v_SI)
    return [DN_Dt, DA_Dt, DCLi_Dt, DCLiO2_Dt]

solution = solve_ivp(residual, [0, time], sol_vec)

nucleations = solution.y[0]
area = solution.y[1]
Li_concentration = solution.y[2]

print(nucleations, Li_concentration)
t = solution.t

plt.figure(0)
plt.plot(t,nucleations)
plt.xlabel("Time (s)")
plt.ylabel("Nucleation (#)")
plt.show()
plt.close()

plt.figure(1)
plt.plot(t,Li_concentration)
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mol m-3)")
plt.show()
plt.close()
"""=========================== Citations ==========================="""

# T. Danner and A. Latz, 2019, 10.1016/j.electacta.2019.134719
# Y. Yin, A. Torayev, C. Gaya, Y. Mammeri, and A. A. Franco, 2017, 10.1021/acs.jpcc.7b05224.

# %%
