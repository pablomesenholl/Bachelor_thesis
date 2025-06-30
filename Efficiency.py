
import numpy as np
from scipy.stats import beta

N_gen_sig = 125885
N_gen_spec = 61142
N_gen_comb = 1200000

N_saved_sig = 10000
N_saved_spec = 20000
N_saved_comb = 6615

N_trig_sig = 28
N_trig_spec = 11
N_trig_comb = 10


def eff(n_total, n_pass):
    return n_pass / n_total


def eff_err(n_total, n_pass):
    efficiency = eff(n_total, n_pass)
    return np.sqrt(efficiency * (1 - efficiency) / n_total)

def clopper_pearson(n_total, n_passed, cl):
    alpha = 1 - cl

    # Lower and upper bounds (Clopper–Pearson)
    lower = beta.ppf(alpha / 2, n_passed, n_total - n_passed + 1)
    upper = beta.ppf(1 - alpha / 2, n_passed + 1, n_total - n_passed)
    return lower, upper

def main():
    eff_gen_sig = eff(N_gen_sig, N_saved_sig)
    eff_gen_spec = eff(N_gen_spec, N_saved_spec)
    eff_gen_comb = eff(N_gen_comb, N_saved_comb)

    eff_gen_sig_err = eff_err(N_gen_sig, N_saved_sig)
    eff_gen_spec_err = eff_err(N_gen_spec, N_saved_spec)
    eff_gen_comb_err = eff_err(N_gen_comb, N_saved_comb)

    cl = 0.6827
    eff_gen_sig_clopper = clopper_pearson(N_gen_sig, N_saved_sig, cl)
    eff_gen_spec_clopper = clopper_pearson(N_gen_spec, N_saved_spec, cl)
    eff_gen_comb_clopper = clopper_pearson(N_gen_comb, N_saved_comb, cl)

    eff_trig_sig = eff(N_saved_sig, N_trig_sig)
    eff_trig_spec = eff(N_saved_spec, N_trig_spec)
    eff_trig_comb = eff(N_saved_comb, N_trig_comb)

    eff_trig_sig_err = eff_err(N_saved_sig, N_trig_sig)
    eff_trig_spec_err = eff_err(N_saved_spec, N_trig_spec)
    eff_trig_comb_err = eff_err(N_saved_comb, N_trig_comb)

    eff_trig_sig_clopper = clopper_pearson(N_saved_sig, N_trig_sig, cl)
    eff_trig_spec_clopper = clopper_pearson(N_saved_spec, N_trig_spec, cl)
    eff_trig_comb_clopper = clopper_pearson(N_saved_comb, N_trig_comb, cl)

    print(f"Gen Efficiency of signal: {eff_gen_sig:4f} +- {eff_gen_sig_err:4f}, upper: {eff_gen_sig_clopper[1] - eff_gen_sig:4f}, lower: {eff_gen_sig - eff_gen_sig_clopper[0]:4f}")
    print(f"Gen Efficiency of spec: {eff_gen_spec:4f} +- {eff_gen_spec_err:4f}, upper: {eff_gen_spec_clopper[1] - eff_gen_spec:4f}, lower: {eff_gen_spec - eff_gen_spec_clopper[0]:4f}")
    print(f"Gen Efficiency of comb: {eff_gen_comb:5f} +- {eff_gen_comb_err:5f}, upper: {eff_gen_comb_clopper[1] - eff_gen_comb:5f}, lower: {eff_gen_comb - eff_gen_comb_clopper[0]:5f}")

    print(f"Trigger Efficiency of signal: {eff_trig_sig:4f} +- {eff_trig_sig_err:4f}, upper: {eff_trig_sig_clopper[1] - eff_trig_sig:4f}, lower: {eff_trig_sig - eff_trig_sig_clopper[0]:4f}")
    print(f"Trigger Efficiency of spec: {eff_trig_spec:4f} +- {eff_trig_spec_err:4f}, upper: {eff_trig_spec_clopper[1] - eff_trig_spec:4f}, lower: {eff_trig_spec - eff_trig_spec_clopper[0]:4f}")
    print(f"Trigger Efficiency of comb: {eff_trig_comb:5f} +- {eff_trig_comb_err:5f}, upper: {eff_trig_comb_clopper[1] - eff_trig_comb:5f}, lower: {eff_trig_comb - eff_trig_comb_clopper[0]:5f}")

if __name__ == '__main__':
    main()