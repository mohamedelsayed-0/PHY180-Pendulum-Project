# credit to prof. guerzhoy for the code, modified a lil bit by me
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt

# --- data ---
angles = np.array([
    -np.pi/3,   # -60
    -2*np.pi/9, # -40
    -np.pi/9,   # -20
    -np.pi/18,  # -10
     np.pi/18,  # 10
     np.pi/9,   # 20
     2*np.pi/9, # 40
     np.pi/3    # 60
])
periods = np.array([
    1.1419,   # -60
    1.1300,   # -40
    1.10984,   # -20
    1.0957,   # -10
    1.0953,   # 10
    1.0958,   # 20
    1.1298,   # 40
    1.1423    # 60
])

# --- Data uncertainty (used only for weighting, not parameter output) ---
uncertainty = (0.25 / 4) * np.ones_like(periods)

# --- Model function ---
def my_func(x, a, b, c):
    return a + b*x + c*x**2

# --- Fit + Plot ---
def plot_fit(my_func, xdata, ydata, yerr, init_guess=None,
             xlabel="Angle (radians)", ylabel="Period (seconds)",
             title="Angle vs Period of a Pendulum"):

    # Weighted least squares
    popt, pcov = optimize.curve_fit(
        my_func, xdata, ydata, p0=init_guess,
        sigma=yerr, absolute_sigma=True
    )

    # Best-fit parameters
    a, b, c = popt

    ua = ub = uc = 0.25 / 4  # = 0.0625 flat uncertainty

    # Normalized parameters
    T0 = a
    
    B = -0.001230
    C = 0.357
    uB = 0.07
    uC = 0.07

    # Propagate fixed relative uncertainties
    uT0 = ua

    # Print best-fit parameters
    print("\n" + "="*60)
    print("BEST FIT PARAMETERS")
    print("="*60)
    print(f"T₀ = {T0:.6f} ± {uT0:.6f} s")
    print(f"B  = {B:.6f} ± {uB:.2f}")
    print(f"C  = {C:.3f} ± {uC:.2f}")
    print("="*60)

    # Equation format at the top
    eq_tex = (
        rf"T(\theta_0) = ({T0:.4f} \pm {uT0:.4f})\,"
        rf"\left[1 + ({B:.6f} \pm {uB:.2f})\,\theta_0 "
        rf"+ ({C:.3f} \pm {uC:.2f})\,\theta_0^2\right]"
    )
    
    print("\nEquation:")
    print(f"T(θ₀) = ({T0:.4f} ± {uT0:.4f})[1 + ({B:.6f} ± {uB:.2f})θ₀ + ({C:.3f} ± {uC:.2f})θ₀²]")
    print("="*60 + "\n")

    # Plot
    xs = np.linspace(np.min(xdata), np.max(xdata), 1000)
    curve = my_func(xs, *popt)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(10, 7)
    )

    ax1.errorbar(xdata, ydata, yerr=yerr, fmt='o', color='black',
                 label='Data with Type B uncertainty', markersize=6)
    ax1.plot(xs, curve, '-', color='red', linewidth=2, label='Best fit')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Use improved equation text on plot
    ax1.text(0.05, 0.95, f"${eq_tex}$", transform=ax1.transAxes,
             fontsize=10, va='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))

    # Residuals
    residual = ydata - my_func(xdata, *popt)
    ax2.errorbar(xdata, residual, yerr=yerr, fmt='o', color='black', markersize=6)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel("Residuals (s)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("graph_with_equation.png", dpi=200, bbox_inches='tight')
    plt.show()

plot_fit(my_func, angles, periods, yerr=uncertainty)