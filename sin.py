import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
from numpy.fft import rfft, rfftfreq

# Load data
data = np.loadtxt('angles.txt', comments="#")
x = data[:, 0]   # time (s)
y = data[:, 1]   # angle (rad)

# Type B uncertainties
time_unc = 0.01    # s
angle_unc = 0.005  # rad

# Damped oscillator model (with optional offset C for robustness)
def damped_cos(t, A, gamma, omega, phi, C):
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi) + C

# Initial parameter guesses
C0 = np.mean(y[-max(5, len(y)//10):])   # baseline from tail
y_detr = y - C0
A0 = 0.5 * (np.max(y_detr) - np.min(y_detr))
A0 = np.abs(A0) if np.isfinite(A0) and A0 > 0 else np.std(y_detr) * np.sqrt(2)

dt = np.median(np.diff(x))
freqs = rfftfreq(len(y_detr), dt)
Y = np.abs(rfft(y_detr))
f0 = freqs[np.argmax(Y[1:])+1] if len(Y) > 1 else 1.0 / max(x[-1]-x[0], 1e-6)
omega0 = 2*np.pi*f0
gamma0 = 0.05
phi0 = 0.0
p0 = [A0, gamma0, omega0, phi0, C0]

# best fit
sigma = angle_unc * np.ones_like(y)
popt, pcov = curve_fit(
    damped_cos, x, y, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=100000
)
A, gamma, omega, phi, C = popt

theta0 = A
tau = np.inf if gamma <= 0 else 1.0/gamma
T = np.inf if omega <= 0 else 2*np.pi/omega
phi0 = phi

# how good the fit is (R^2)
yhat = damped_cos(x, *popt)
ss_res = np.sum((y - yhat)**2)
ss_tot = np.sum((y - np.mean(y))**2)
R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

# create fitted curve
tfit = np.linspace(x.min(), x.max(), 2000)
yfit = damped_cos(tfit, *popt)

# plot
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
})

fig, ax = plt.subplots(figsize=(10, 6))

# data points
ax.scatter(x, y, s=18, color='#d62728', edgecolor='white', linewidth=0.6, alpha=0.9, label='Data')

# fit curve
ax.plot(tfit, yfit, linewidth=2.5, linestyle='--', color='#1f77b4', label='Damped fit')

# uncertainity band (pm tracker angle uncertainty around the fit)
ax.fill_between(tfit, yfit - angle_unc, yfit + angle_unc, alpha=0.15, color='#1f77b4',
                label='± angle uncertainty')

# labels, ticks
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (rad)')
ax.set_title('Time vs. Angle')

ax.grid(True, which='major', linestyle='--', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.35)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

for spine in ax.spines.values():
    spine.set_alpha(0.6)

# margins
ax.margins(x=0.02, y=0.08)

# Legend
ax.legend(frameon=True, framealpha=0.9)

# text for eq
eq_symbolic = r"$\theta(t)=\theta_0\,e^{-t/\tau}\cos\!\left(2\pi\,\frac{t}{T}+\phi_0\right)$"
eq_numeric  = rf"$\theta(t)={theta0:.4g}\,e^{{-t/{tau:.4g}}}\cos\!\left(2\pi\,\frac{{t}}{{{T:.4g}}}+{phi0:.4g}\right)$"

lines = [eq_symbolic, eq_numeric]
if abs(C) > 5e-4:   # show only if meaningful
    lines.append(rf"(offset: $C={C:.3g}$ rad)")
lines.append(rf"$R^2={R2:.4f}$")

ax.text(
    0.98, 0.98, "\n".join(lines),
    transform=ax.transAxes,
    fontsize=11, va='top', ha='right',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.92, edgecolor='#c0c0c0')
)

fig.tight_layout()
plt.show()

# terminal printout
print("Displayed equation form:")
print("  θ(t) = θ0 e^{-t/τ} cos( 2π t / T + φ0 )")
print(f"  θ0 = {theta0:.6g} rad,   τ = {tau:.6g} s,   T = {T:.6g} s,   φ0 = {phi0:.6g} rad,   offset C = {C:.6g} rad,   R^2 = {R2:.6f}")
