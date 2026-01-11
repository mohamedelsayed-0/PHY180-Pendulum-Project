import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
L = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
T = np.array([0.769, 0.895, 1.158, 1.337, 1.545, 1.698, 1.789, 1.9091, 2.0, 2.014], dtype=float)

# Uncertainties (
period_uncertainty = 0.125   # s
length_uncertainty = 0.001   # m

# imposed upon model: T = k * L^n
def power_law(x, k, n):
    return k * x**n

popt, pcov = curve_fit(
    power_law, L, T,
    sigma=np.full_like(T, period_uncertainty, dtype=float),
    absolute_sigma=True,         # treat sigma as true 1σ
    p0=(2.0, 0.5)                # sensible start near sqrt(L)
)

k_fit, n_fit = popt
k_uncertainty, n_uncertainty = np.sqrt(np.diag(pcov))

# sig fig rounding as per instructions
def round_uncertainty_to_1sf(u):
    if u == 0 or not np.isfinite(u):
        return 0.0, 0
    p = int(np.floor(np.log10(abs(u))))   # order of magnitude
    d = -p                                # decimals needed for 1 significant figure
    u_rounded = np.round(u, d)
    decimals = max(d, 0)
    return float(u_rounded), int(decimals)

def round_value_to_decimals(x, decimals):
    return float(np.round(x, decimals))

# k
k_unc_1sf, k_dec = round_uncertainty_to_1sf(k_uncertainty)
k_val_rounded = round_value_to_decimals(k_fit, k_dec)

# n
n_unc_1sf, n_dec = round_uncertainty_to_1sf(n_uncertainty)
n_val_rounded = round_value_to_decimals(n_fit, n_dec)

k_str = f"{k_val_rounded:.{k_dec}f} ± {k_unc_1sf:.{k_dec}f}"
n_str = f"{n_val_rounded:.{n_dec}f} ± {n_unc_1sf:.{n_dec}f}"

# smooth curve for plotting
L_fit = np.linspace(L.min(), L.max(), 300)
T_fit = power_law(L_fit, k_fit, n_fit)

#error bars
plt.figure(figsize=(8, 6))

plt.errorbar(
    L, T,
    xerr=length_uncertainty,
    yerr=period_uncertainty,
    fmt='o',
    color='blue',
    ecolor='black',
    elinewidth=1.5,
    capsize=5,
    capthick=1.5,
    label='Measured Data (± listed uncertainties)'
)

plt.plot(
    L_fit, T_fit, 'r-', linewidth=2,
    label=f'Fit: T = ({k_str}) · L^({n_str})'
)

plt.title('Pendulum Period vs. Length with Listed Uncertainties', fontsize=14)
plt.xlabel('Length (m)', fontsize=12)
plt.ylabel('Period (s)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#results
print(f"k = {k_str}")
print(f"n = {n_str}")
