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
Rbar = 8.3145 #J mol-1 K-1 // Ideal constant

"""======================== User system variables ========================="""
#Parameters that can be changed during the trials
T = 25 + 273.15 #K // Temperature
C_LiO2_0 = 0.15 #mol m-3 // Electrolyte Concentration of LiO2 particles
C_Li_0 = 0.15 #mol m-3 // Electrolyte Concentration Li+ particles
Elyte_v = 1E-10 #uL3 // Electrolyte Volume
N_0 = 0 # // Nucleations
R_0 = 0
A_0 = m.pi*0.009**2 #m2 // Cross sectional area
time = 5000 #s // simulation time
atol = 1e-8 # absolute tolerance
rtol = 1e-4 # relative tolerance

"""======================== User thermodyanic imputs ========================"""
#parameters that depend on the system being studied

C_LiO2_sat = 0.1 #mol m-3 // saturated concentration of LiO2 //Yin (2017)
C_Li_sat = 0.1 #mol m-3 // saturated concentration of LiO2
k_surf = 1E-8 #mol s-1 m-2 // kinetic rate constant for the growth reaction // Yin (2017)
k_surf_des = 1E-8 #mol s-1 m-2 // kinetic rate constant for desorption // Yin (2017)
gamma_surf = 7.7E-2 #J m-2 //  surface energy of the newly formed crystal phase// Danner (2019) - should be replaced for LiS system
sig_surf = 0.75 # J m-2 // Specific surface energy of Li2O2 with the electrolyte // Yin (2017)
V = 1.98E-5 #m3 mol-1 // molar volume// Yin (2017)
theta = 30*m.pi/180 # radians // Contact angle // Danner (2019) - should be replaced for LiS system
D_LiO2 = 1.2E-9 #m2 -s-1 // diffusion ceofficient // Yin (2017)


"""======================== Thermodyanic system cals ========================"""
Elyte_v_SI = Elyte_v*1.0E-9 #
phi = (2+m.cos(theta))*(1-m.cos(theta))**2/4  # - // contact angle correction factor

"""========================== Initialization step =========================="""
#%%
#intializing the solution vector
initial = {}
initial['A'] = A_0
initial['C_Li'] = C_Li_0
initial['C_LiO2'] = C_LiO2_0
sol_vec_1 = list(initial.values())  # solution vector

"""=========================== Equations ==========================="""
#%%
radii = np.linspace(1e-10, 1e-6, 200)
bin_width = radii[2]-radii[1]
Nucleation_con = np.zeros_like(radii)

sol_vec = np.hstack([sol_vec_1, Nucleation_con])

data_nuc = pd.DataFrame(index = radii)

def residual(t, sol_vec):
    A, C_Li, C_LiO2 = sol_vec[:3]
    n_p = sol_vec[3:]
    Dnp_dt = np.zeros_like(n_p)
    Dr_dt = np.zeros_like(n_p)
    a_d = (C_LiO2*N_a)**(-1/3) # length scale of diffusion
    r_crit = 2*gamma_surf*V/(Rbar*T*m.log(C_LiO2/C_LiO2_sat*C_Li/C_Li_sat)) # m // critical radius
    N_crit = 4/3*m.pi*r_crit**3*N_a/V # number of molecules in the critical nucleus of size
    Del_G_Crit = phi*4/3*m.pi*gamma_surf*r_crit**2 # J mol-1 // energy barrier of the nucleation
    if N_crit <0:
        Del_G_Crit =0
    Z = m.sqrt(Del_G_Crit/(phi*3*m.pi*k_B*T*N_crit)) # - // Zeldovich factor
    V_crit = 4/3*m.pi*r_crit**3 # m3 // Critical volume
    N_sites = A/(m.pi*r_crit**2) # number of nucleation sites
    k_nuc= D_LiO2*(a_d**-2) #nucleation rate calculated based on the distance between particles
    DN_Dt = k_nuc*N_sites*Z*m.exp(-Del_G_Crit/(k_B*T))*1E-20
    for i, r in enumerate(radii):
        if r > r_crit:
            Dnp_dt[i] += DN_Dt
            break
    for i, N in enumerate(n_p):
        Dr_dt[i] = D_LiO2*V*(C_Li- C_Li_sat)*(C_LiO2-C_LiO2_sat)/(radii[i]+D_LiO2/k_surf)- m.pi*radii[i]**2*N*gamma_surf*k_surf_des
        dNdt_radii = Dr_dt[i]/bin_width*N
        if dNdt_radii <0:
            Dnp_dt[i] += dNdt_radii
            if i > 0:
                Dnp_dt[i-1] -= dNdt_radii
        elif dNdt_radii > 0 and radii[i] != radii[-1]:
            Dnp_dt[i] -= dNdt_radii
            Dnp_dt[i+1] += dNdt_radii
    data_nuc[C_Li] = Dnp_dt.tolist()
    DCLi_Dt = -DN_Dt*V_crit/(V*Elyte_v_SI) - 2*np.sum(Dr_dt*radii*radii)*np.pi*V_crit/(V*Elyte_v_SI)
    DCLiO2_Dt = -DN_Dt*V_crit/(V*Elyte_v_SI)- 2*np.sum(Dr_dt*radii*radii)*np.pi*V_crit/(V*Elyte_v_SI)
    DA_Dt = - DN_Dt*m.pi*r_crit**2 - 4*np.pi*np.sum(radii*Dr_dt)*np.pi
    sol_vectemp = [DA_Dt, DCLi_Dt, DCLiO2_Dt]
    bundle = np.hstack([sol_vectemp, Dnp_dt])
    return bundle

solution = solve_ivp(residual, [0, time], sol_vec, method='BDF',
        rtol=rtol, atol=atol)
#%%
data_nuc.to_csv(r'C:\Users\Mels\Code\HNG-model\output.csv')
Area = solution.y[0]
Concentration_Li = solution.y[1]
Concentration_LiO2 = solution.y[2]
histograms = solution.y[3:]
t1 = histograms[:,0]
tmid = histograms[:,18]
t2 = histograms[:,-1]

plt.figure(3)
plt.plot(radii, t1)
plt.xlabel("Radius (s)")
plt.ylabel("Nuclii (#)")


plt.figure(5)
plt.plot(radii, tmid)
plt.xlabel("Radius (s)")
plt.ylabel("Nuclii (#)")



plt.figure(4)
plt.plot(radii, t2)
plt.xlabel("Radius (s)")
plt.ylabel("Nuclii (#)")


#%%
t = solution.t

plt.figure(0)
plt.plot(t,Area)
plt.xlabel("Time (s)")
plt.ylabel("A (m2)")
plt.xlim( [0.00004, 0.00009])

plt.figure(1)
plt.plot(t,Concentration_Li)
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mol m-3)")
plt.show()
plt.close()

"""=========================== Citations ==========================="""

# T. Danner and A. Latz, 2019, 10.1016/j.electacta.2019.134719
# Y. Yin, A. Torayev, C. Gaya, Y. Mammeri, and A. A. Franco, 2017, 10.1021/acs.jpcc.7b05224.
# Official soundtrack: Mamma Mia!/Mamma Mia! Here We Go Again: The Movie Soundtracks Featuring Songs of ABBA
# %%
