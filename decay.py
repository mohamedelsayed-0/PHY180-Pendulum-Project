import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Load data
file_path = "angles.txt"
data = np.loadtxt(file_path, comments="#", skiprows=1)  # skip header if exists

t = data[:, 0]   # time
x = data[:, 1]   # displacement

# Uncertainty in tau from tracker (30 fps → 1/30 s)
tau_unc = 1/30   # seconds ≈ 0.033 s

# Find peaks (amplitudes)
peaks, _ = find_peaks(x)
t_peaks = t[peaks]
x_peaks = x[peaks]

# Define exponential decay function
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

# Fit exponential to the peaks
popt, _ = curve_fit(exp_decay, t_peaks, x_peaks, p0=(x_peaks[0], 1, 0))
A, tau, C = popt

# generate smooth fit curve -
t_fit = np.linspace(min(t_peaks), max(t_peaks), 2000)
x_fit = exp_decay(t_fit, *popt)

# threshold at 4% of max amplitude (e^-pi)
max_amp = np.max(x_peaks)
threshold = 0.04 * max_amp

# Count how many peaks are above threshold
num_peaks_before_threshold = np.sum(x_peaks > threshold)

# Plot (with tau uncertainty shown on y-axis)
plt.figure(figsize=(10, 6))

# Fit line
plt.plot(t_fit, x_fit, 'r-', linewidth=2.2, alpha=0.9,
         label=f"Fit: y = {A:.3f} * exp(-t/({tau:.3f} ± {tau_unc:.3f})) + {C:.3f}")

# peaks
plt.errorbar(t_peaks, x_peaks, yerr=tau_unc, fmt='o', 
             color="navy", ecolor="blue", elinewidth=1, capsize=2, alpha=0.7,
             markerfacecolor="blue", markeredgewidth=1.2, label="Peaks (±Δτ)")

# q factor threshold
plt.axhline(threshold, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, 
            label="4% Threshold")

# labels and styles
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Amplitude (Rad)", fontsize=12)
plt.title("Exponential Decay Fit of Peaks", fontsize=14, weight="bold")
plt.legend(frameon=True, fontsize=10, loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)

# zoom out
plt.ylim(min(x_peaks) - 0.1*max_amp, max_amp * 1.1)

plt.show()

#  results
print(f"Exponential decay fit: y = {A:.6f} * exp(-t/({tau:.6f} ± {tau_unc:.6f})) + {C:.6f}")
print(f"Tracker-based tau uncertainty: Δτ = {tau_unc:.6f} s (from 30 fps)")
print(f"Number of amplitudes (peaks) found: {len(x_peaks)}")
print(f"Number of peaks before dropping below 4%: {num_peaks_before_threshold}")
