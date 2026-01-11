import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# data
L = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) 
T = np.array([0.769, 0.895, 1.158, 1.337, 1.545, 1.698, 1.789, 1.9091, 2.0, 2.014])

# log both sides
logL = np.log10(L)
logT = np.log10(T)

# linear regression
slope, intercept, r_value, p_value, std_err = linregress(logL, logT)
n = slope
k = 10 ** intercept
r2 = r_value ** 2

print(f"n = {n:.3f}")
print(f"k = {k:.3f}")
print(f"R² = {r2:.4f}")

# uncertaintes
dk = 0.07     # uncertainty in k
dn = 0.05     # uncertainty in n

# vertical error in log-space from change in k
const_yerr = dk / (k * np.log(10))

# best fit line
logL_fit = np.linspace(logL.min(), logL.max(), 200)
logT_fit = intercept + slope * logL_fit

#  plot
plt.figure(figsize=(10, 7))

plt.errorbar(
    logL, logT,
    yerr=const_yerr,
    fmt='o',
    color='blue',
    ecolor='black',
    elinewidth=1.5,
    capsize=5,
    capthick=1.5,
    label='Measured Data with uncertaintes'
)

plt.plot(
    logL_fit, logT_fit,
    color='red',
    linewidth=2,
    label=f'Best Fit: log(T) = {n:.2f} log(L) + {intercept:.2f}'
)

plt.xlabel('log₁₀(L) (unitless)', fontsize=14)
plt.ylabel('log₁₀(T) (unitless)', fontsize=14)
plt.title('Log–Log Plot of Pendulum Period vs Length', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
