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
pi = 3.141529
N_a = 6.022E23 #mol-1 // Avogadro's number
R = 8.3145 #J mol-1 K-1 // Ideal constant

"""=========================== User system variables ==========================="""
#Parameters that can be changed during the trials
T = 25 + 273.15 #K
C_LiO2 = 0.15 #mol m-3
C_Li = 0.15 #mol m-3
Elyte_v = 0.0005 #m3
N_0 = 0
A_0 = pi*0.009**2 #m2
time = 100000

"""=========================== User thermodyanic imputs ==========================="""
#parameters that depend on the system being studied

C_LiO2_sat = 0.1 #mol m-3 // Yin (2017)
C_Li_sat = 0.1
k_nu = 1E-6 #mol s-1 m-2 // Yin (2017)
gamma_surf = 7.7E-3 #J m-2 // Danner (2019) - should be replaced for LiS system
V = 1.98E-5 #m3 mol-1 // Yin (2017)
theta = 30*pi/180 # radians // Danner (2019) - should be replaced for LiS system
D_LiO2 = 1.2E-9 #m2 -s-1 // Yin (2017)


"""=========================== Thermodyanic system cals ==========================="""

a_d = (C_LiO2*N_a)**(-1/3)
phi = (2+m.cos(theta))*(1-m.cos(theta))**2/4
r_crit = 2*gamma_surf*V/(R*T*m.log(C_LiO2/C_LiO2_sat*C_Li/C_Li_sat))
N_crit = 4/3*pi*r_crit**3*N_a/V
Del_G_Crit = phi*4/3*pi*gamma_surf*r_crit**2
Z = m.sqrt(Del_G_Crit/(phi*3*pi*k_B*T*N_crit))

N_sites_0 =  A_0/(pi*r_crit**2)

"""=========================== Initialization step ==========================="""
#%%
#intializing the solution vector
initial = {}
initial['N'] = N_0
initial['N_sites'] = N_sites_0
sol_vec = list(initial.values())  # solution vector
print(sol_vec)

"""=========================== Equations ==========================="""
#%%


def residual(t, solution):
    N, N_sites = solution
    DN_Dt = D_LiO2*a_d*N_sites*Z*m.exp(-Del_G_Crit/(k_B*T))
    DN_sites_Dt = - DN_Dt
    return [DN_Dt, DN_sites_Dt]

solution = solve_ivp(residual, [0, time], sol_vec)

nucleations = solution.y[0]
print(nucleations)
t = solution.t

plt.figure(0)
plt.plot(t,nucleations)
plt.xlabel("Time (s)")
plt.ylabel("Nucleation (#)")
plt.show()
plt.close()
"""=========================== Citations ==========================="""

# T. Danner and A. Latz, 2019, 10.1016/j.electacta.2019.134719
# Y. Yin, A. Torayev, C. Gaya, Y. Mammeri, and A. A. Franco, 2017, 10.1021/acs.jpcc.7b05224.

# %%
