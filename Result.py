# try CLs method
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt


def cls_counting(nobs, s, b):
    p_sb = poisson.cdf(nobs, s + b)  # P(N ≥ nobs | s+b)
    p_b = poisson.cdf(nobs, b)  # P(N ≥ nobs |  b )
    return 1.0 if p_b == 0 else p_sb / p_b

# branching ratios
BR_tau_hadr = 0.1825 #BR of TAUOLA 5
BR_tau_muonic = 0.1739 #BR of TAUOLA 2
BR_Kstar = 2.0/3.0 #BR of Kstar to K^+ pi^-
BR_eff_sig = 2 * BR_tau_hadr * BR_tau_muonic * BR_Kstar
sigma_BR_eff = 0.000277332


# inputs
eff_gen_comb = 6615/1200000
eff_trigger_comb = 1.512e-3
eff_cls_comb = 0.009064422143088377
L = 40e3
sigma_bb = 4.467e08
sigma_sigma_bb = 191189040
sigma_cc = 8.004e09
sigma_tot = sigma_bb + sigma_cc
N_comb_bckg = L * sigma_tot * eff_trigger_comb * eff_gen_comb * eff_cls_comb
# N_bckg = 25538371.046163805
eff0, sigma_eff0 = 9.57581e-5, 2.00059e-5
b0, sigma_b = N_comb_bckg, 32147474.34
n_toys = 10000
br_grid = np.logspace(-6, -3, 500)  # ascending

cls_toys = np.empty((n_toys, len(br_grid)), dtype=float)
limits = []
for i in range(n_toys):
    b_alt = np.random.lognormal(np.log(b0/(np.sqrt(b0**2 + sigma_b**2))), np.sqrt(np.log(1 +  (sigma_b / b0)**2)))
    eff_alt = np.random.lognormal(np.log(eff0/(np.sqrt(eff0**2 + sigma_eff0**2))), np.sqrt(np.log(1 + (sigma_eff0 / eff0)**2)))
    sigma_bb_alt = np.random.lognormal(np.log(sigma_bb/(np.sqrt(sigma_bb**2 + sigma_sigma_bb**2))), np.sqrt(np.log(1 + (sigma_sigma_bb/sigma_bb)**2)))
    BR_eff_sig_alt = np.random.lognormal(np.log(BR_eff_sig/(np.sqrt(BR_eff_sig**2 + sigma_BR_eff**2))), np.sqrt(np.log(1 + (sigma_BR_eff/BR_eff_sig)**2)))

    # expected-median option (Asimov data); replace with np.random.poisson if wanted
    n_obs = int(round(b_alt))
    s_grid = L * sigma_bb_alt * eff_alt * br_grid * BR_eff_sig_alt
    cls_toys[i] = cls_counting(n_obs, s_grid, b_alt)

    for br in br_grid:
        s = L * sigma_bb_alt * eff_alt * br * BR_eff_sig_alt
        if cls_counting(n_obs, s, b_alt) < 0.05:
            limits.append(br)
            break

limits = np.asarray(limits)
br_95 = limits.mean()
br_95_err = limits.std(ddof=1)

print(f"Expected 95% CL upper limit: BR = {br_95:.2e} ± {br_95_err:.2e}")

lim_median = np.quantile(limits, 0.50)
lim_1lo = np.quantile(limits, 0.16)  # –1 σ
lim_1hi = np.quantile(limits, 0.84)  # +1 σ
lim_2lo = np.quantile(limits, 0.025)  # –2 σ
lim_2hi = np.quantile(limits, 0.975)  # +2 σ

print("\n95 % CL expected upper limit (to compare with mean±σ):")
print(f"  median : {lim_median: .2e}")
print(f"  –1σ    : {lim_median - lim_1lo   : .2e}")
print(f"  +1σ    : {lim_1hi - lim_median   : .2e}")
# print(f"  –2σ    : {lim_2lo   : .2e}")
# print(f"  +2σ    : {lim_2hi   : .2e}\n")


median_exp = np.percentile(cls_toys, 50, axis=0)
band1_lo = np.percentile(cls_toys, 16, axis=0)
band1_hi = np.percentile(cls_toys, 84, axis=0)
band2_lo = np.percentile(cls_toys, 2.5, axis=0)
band2_hi = np.percentile(cls_toys, 97.5, axis=0)


def first_crossing(cls_curve, threshold=0.05):
    idx = np.argmax(cls_curve < threshold)  # first True
    return br_grid[idx] if cls_curve[idx] < threshold else np.nan


# ---------- OBSERVED CURVE ----------
# rebuild cls with the real data
s_grid_obs = L * sigma_bb * eff0 * br_grid * BR_eff_sig
cls_obs = cls_counting(N_comb_bckg, s_grid_obs, b0)

# ---------- PLOT ----------
fig, ax = plt.subplots()
ax.fill_between(br_grid, band2_lo, band2_hi,
                step="mid", alpha=0.5, label=r"Expected $\pm2\sigma$")
ax.fill_between(br_grid, band1_lo, band1_hi,
                step="mid", alpha=0.9, label=r"Expected $\pm1\sigma$")
ax.plot(br_grid, median_exp, lw=1.5, linestyle="--",
        label="Median")
ax.axhline(0.05, color="black", lw=1.0)
ax.text(br_grid[1], 0.055, "95% CL", va="bottom", ha="left")
ax.set_xscale("log")
ax.set_xlim(br_grid[0], br_grid[-1])
ax.set_ylim(0, 1.1)
ax.set_ylabel(r"$\alpha'$")
ax.set_xlabel(r"Branching ratio $\,\mathcal{B}$")
ax.grid(True, which="both", ls=":")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.savefig("Result_plots/CLs_vs_BR_result.png")