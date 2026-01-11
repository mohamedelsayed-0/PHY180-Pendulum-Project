import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# data
L = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], dtype=float)
T = np.array([0.769, 0.895, 1.158, 1.337, 1.545, 1.698, 1.789, 1.9091, 2.0, 2.014], dtype=float)

# uncertaintes
sigma_Q = 2.0            # Q (dimensionless)
sigma_L = 0.001          # length uncertainty (m) / = 0.1 cm

tau = 33.350             # dimensionless
Q = np.pi * tau / T      # dimensionless 

# try different models Q(L) — ODR with sx = sigma_L and sy = sigma_Q
def linear(B, x):        # B=[m,b]
    m, b = B
    return m*x + b

def powerlaw(B, x):      # B=[a,n]
    a, n = B
    return a * x**n

def exponential(B, x):   # B=[A,B]
    A, Bp = B
    return A * np.exp(Bp * x)

def reciprocal(B, x):    # B=[a,b]
    a, b = B
    return a / x + b

candidates = [
    ("Linear",     Model(linear),      np.array([(Q[-1]-Q[0])/(L[-1]-L[0]), Q.mean()])),
    ("Power law",  Model(powerlaw),    np.array([Q[0]/(L[0]**0.5), 0.5])),
    ("Exponential",Model(exponential), np.array([Q[0], np.log(Q[-1]/Q[0])/(L[-1]-L[0]) ])),
    ("Reciprocal", Model(reciprocal),  np.array([Q[0]*L[0], 0.0]))
]

def fit_model(name, model, beta0):
    data = RealData(L, Q, sx=np.full_like(L, sigma_L, dtype=float), sy=np.full_like(Q, sigma_Q, dtype=float))
    odr  = ODR(data, model, beta0=beta0)
    out  = odr.run()
    n = len(L)
    k = len(out.beta)
    chi2 = out.sum_square            
    dof  = max(n - k, 1)
    red_chi2 = chi2 / dof
    AIC  = 2*k + chi2
    AICc = AIC + (2*k*(k+1)) / (n - k - 1) if (n - k - 1) > 0 else np.inf
    return {"name": name, "beta": out.beta, "sd_beta": out.sd_beta,
            "red_chi2": red_chi2, "AICc": AICc, "fcn": model.fcn}

results = [fit_model(nm, mdl, b0) for nm, mdl, b0 in candidates]
best = min(results, key=lambda r: r["AICc"])

def eqn_string(res):
    name = res["name"]
    b, db = res["beta"], res["sd_beta"]
    if name == "Linear":
        m, c = b; dm, dc = db
        return f"Q(L) = ({m:.2f} ± {dm:.2f})·L + ({c:.2f} ± {dc:.2f})"
    if name == "Power law":
        a, n = b; da, dn = db
        return f"Q(L) = ({a:.2f} ± {da:.2f})·L^({n:.2f} ± {dn:.2f})"
    if name == "Exponential":
        A, Bp = b; dA, dBp = db
        return f"Q(L) = ({A:.2f} ± {dA:.2f})·exp[({Bp:.3f} ± {dBp:.3f})·L]"
    if name == "Reciprocal":
        a, c = b; da, dc = db
        return f"Q(L) = ({a:.2f} ± {da:.2f})/L + ({c:.2f} ± {dc:.2f})"
    return "Model not recognized."

print(f"Selected model: {best['name']}")
print(f"Reduced χ² = {best['red_chi2']:.3f}")
print(eqn_string(best))

#  Plot: Q vs L
L_line = np.linspace(L.min(), L.max(), 400)
Q_line = best["fcn"](best["beta"], L_line)

plt.figure(figsize=(10, 7))
plt.errorbar(
    L, Q, xerr=sigma_L, yerr=sigma_Q,
    fmt='o', color='tab:blue',
    ecolor='black', elinewidth=1.5, capsize=5, capthick=1.5,
    label='Q data (±2 in Q, ±0.001 m in L)'
)
plt.plot(L_line, Q_line, 'r-', linewidth=2,
         label=f'{best["name"]} fit:\n{eqn_string(best)}')
plt.title("Q Factor vs. Length")
plt.xlabel("Length (m)")
plt.ylabel("Q Factor")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
