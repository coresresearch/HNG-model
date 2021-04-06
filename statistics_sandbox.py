#%%
from scipy.stats import lognorm
import numpy as np
import math as m
import matplotlib.pyplot as plt

tester = np.array([1,2,3])
std = np.std(tester)
avg = sum(tester)/len (tester)

mu = m.log(avg**2/m.sqrt(avg**2+std**2))
sigma = m.sqrt(m.log(1+std**2/avg**2))
total = len(tester)
dist = lognorm(total, mu, sigma )
print (dist)
x = np.linspace(0,6,200)

plt.figure(0)
plt.plot(x,dist.cdf(x))
plt.xlabel("Time (s)")
plt.ylabel("Nucleation (#)")

plt.figure(0)
plt.plot(x,dist.pdf(x))
plt.xlabel("Time (s)")
plt.ylabel("Nucleation (#)")

# %%
