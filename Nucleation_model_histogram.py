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
Elyte_v = 1E-8 #uL3 // Electrolyte Volume
A_0 = m.pi*0.009**2 #m2 // Cross sectional area
time = 5000 #s // simulation time
atol = 1e-8 # absolute tolerance
rtol = 1e-4 # relative tolerance
r_min = 1e-10 # m // minimum particle radius to consider
r_max = 1e-6 # m // maximum particle radius to consider
n_bins = 200 # Number of histogram bins for the radii discretization

"""================== User thermodyanic and kinetic imputs =================="""
#parameters that depend on the system being studied

C_LiO2_sat = 0.1 #mol m-3 // saturated concentration of LiO2 //Yin (2017)
C_Li_sat = 0.1 #mol m-3 // saturated concentration of LiO2
k_surf = 1E-4 #mol s-1 m-2 // kinetic rate constant for the growth reaction // Yin (2017)
k_surf_des = 1E-8 #mol s-1 m-2 // kinetic rate constant for desorption // Yin (2017)
gamma_surf = 7.7E-2 #J m-2 //  surface energy of the newly formed crystal phase// Danner (2019) - should be replaced for LiS system
sig_surf = 0.75 # J m-2 // Specific surface energy of Li2O2 with the electrolyte // Yin (2017)
v_Li2O2 = 1.98E-5 #m3 mol-1 // molar volume// Yin (2017)
theta = 30*m.pi/180 # radians // Contact angle // Danner (2019) - should be replaced for LiS system
D_LiO2 = 1.2E-9 #m2 -s-1 // diffusion ceofficient // Yin (2017)


"""======================= Thermodyanic system calcs ========================"""
Elyte_v_SI = Elyte_v*1.0E-9 # Convert to m3
phi = (2+m.cos(theta))*(1-m.cos(theta))**2/4  # - // contact angle correction factor

"""========================== Initialization step =========================="""
#intializing the solution vector
initial = {}
initial['A'] = A_0
initial['C_Li'] = C_Li_0
initial['C_LiO2'] = C_LiO2_0
sol_vec_1 = list(initial.values())  # solution vector

radii = np.linspace(r_min, r_max, n_bins)
bin_width = radii[2]-radii[1]
particle_concentrations = np.zeros_like(radii)

sol_vec = np.hstack([sol_vec_1, particle_concentrations])

"""=========================== Equations ==========================="""

data_nuc = pd.DataFrame(index = radii)

def residual(t, sol_vec):

    # Read out current state:
    A, C_Li, C_LiO2 = sol_vec[:3]
    n_p = sol_vec[3:]

    # Initialize derivatives:
    Dnp_dt = np.zeros_like(n_p)
    Dr_dt = np.zeros_like(n_p)

    # For convenience, store the product Rbar*T:
    RT = Rbar*T
    
    a_d = (C_LiO2*N_a)**(-1./3.) # length scale of diffusion
    
    # m // critical radius
    r_crit = (2. * gamma_surf * v_Li2O2 
        / (RT * m.log(C_LiO2 / C_LiO2_sat * C_Li / C_Li_sat))) 
    
    # number of moles in the nucleus with radius r_crit
    N_crit = 4./3.*m.pi*r_crit**3./v_Li2O2 
    """Is the division by 3 correct? If I'm guessing correctly, this is surface area, 4pi r_crit^2..."""
    # J mol-1 // energy barrier of the nucleation
    dG_crit = phi*4./3.*m.pi*gamma_surf*r_crit**2 
    if N_crit <0:
        dG_crit = 0

    """Make sure the units work out, here."""
    Z = m.sqrt(dG_crit/(phi * 3. * m.pi * RT * N_crit)) # - // Zeldovich factor
    V_crit = 4./3. * m.pi * r_crit**3 # m3 // Critical volume
    """TODO: incorporate max packing fraction for circles."""
    N_sites = A/(m.pi*r_crit**2) # number of nucleation sites
    k_nuc= D_LiO2*(a_d**-2) #nucleation rate calculated based on the distance between particles
    """Should k_B*T be R*T, here, since Del_G_Crit is in J/mol?"""
    DN_Dt = k_nuc*N_sites*Z*m.exp(-dG_crit/RT)*1E-20
    for i, r in enumerate(radii):
        if r > r_crit:
            Dnp_dt[i] += DN_Dt
            break
    for i, N in enumerate(n_p):
        Dr_dt[i] = D_LiO2*v_Li2O2*(C_Li- C_Li_sat)*(C_LiO2-C_LiO2_sat)/(radii[i]+D_LiO2/k_surf)- m.pi*radii[i]**2*N*gamma_surf*k_surf_des
        dNdt_radii = Dr_dt[i]/bin_width*N
        if dNdt_radii <0:
            Dnp_dt[i] += dNdt_radii
            if i > 0:
                Dnp_dt[i-1] -= dNdt_radii
        elif dNdt_radii > 0 and radii[i] != radii[-1]:
            Dnp_dt[i] -= dNdt_radii
            Dnp_dt[i+1] += dNdt_radii
    data_nuc[C_Li] = Dnp_dt.tolist()
    DCLi_Dt = -DN_Dt*V_crit/(v_Li2O2*Elyte_v_SI) - 2.*np.sum(Dr_dt*radii*radii)*np.pi*V_crit/(v_Li2O2*Elyte_v_SI)
    DCLiO2_Dt = -DN_Dt*V_crit/(v_Li2O2*Elyte_v_SI)- 2.*np.sum(Dr_dt*radii*radii)*np.pi*V_crit/(v_Li2O2*Elyte_v_SI)
    DA_Dt = - DN_Dt*m.pi*r_crit**2 - 4.*np.pi*np.sum(radii*Dr_dt)
    dSolVec_dt_temp = [DA_Dt, DCLi_Dt, DCLiO2_Dt]
    dSolVec_dt = np.hstack([dSolVec_dt_temp, Dnp_dt])
    return dSolVec_dt

solution = solve_ivp(residual, [0, time], sol_vec, method='BDF',
        rtol=rtol, atol=atol)
#%%
data_nuc.to_csv(r'C:\Users\Mels\Code\HNG-model\output.csv')
Area = solution.y[0]
Concentration_Li = solution.y[1]
Concentration_LiO2 = solution.y[2]
histograms = solution.y[3:]
n_steps = histograms.shape[1]
particles_0 = histograms[:,0]
particles_mid = histograms[:,int(m.floor(n_steps/2))]
t_mid = solution.t[int(m.floor(n_steps/2))]
particles_final = histograms[:,-1]

plt.figure(3)
plt.plot(radii, particles_0)
plt.xlabel("Radius (s)")
plt.ylabel("Nuclii (#)")
plt.title("Initial particle distribution, t = 0 s.")


plt.figure(5)
plt.plot(radii, particles_mid)
plt.xlabel("Radius (s)")
plt.ylabel("Nuclii (#)")
plt.title(f"Particle distribution at t = {t_mid:.2f} s.")



plt.figure(4)
plt.plot(radii, particles_final)
plt.xlabel("Radius (s)")
plt.ylabel("Nuclii (#)")
plt.title(f"Final particle distribution, t = {solution.t[-1]:.0f} s.")


#%%
t = solution.t

plt.figure(0)
plt.plot(t,Area)
plt.xlabel("Time (s)")
plt.ylabel("A (m2)")
# plt.xlim( [0.00004, 0.00009])

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
