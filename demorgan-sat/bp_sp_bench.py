#!/usr/bin/env python3
"""
Belief Propagation (BP) and Survey Propagation (SP) initialisation benchmark
on SATLIB uf* benchmarks — optimised implementation.
"""
import numpy as np
import os, glob, time

from benchmark_paths import benchmark_root

BENCH_ROOT = benchmark_root()

# ============================================================
# DIMACS parser
# ============================================================
def read_dimacs(path):
    clauses = []
    n_vars = 0
    with open(path) as f:
        buf = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('c') or line.startswith('%'):
                continue
            if line.startswith('p'):
                parts = line.split()
                n_vars = int(parts[2])
                continue
            nums = list(map(int, line.split()))
            for v in nums:
                if v == 0:
                    if buf:
                        clauses.append(buf)
                        buf = []
                else:
                    buf.append(v)
    return n_vars, clauses

# ============================================================
# De Morgan encoding
# ============================================================
def demorgan_encode(n_vars, clauses):
    h = np.zeros(n_vars)
    J_dict = {}
    for clause in clauses:
        l1, l2, l3 = clause[0], clause[1], clause[2]
        v1, v2, v3 = abs(l1)-1, abs(l2)-1, abs(l3)-1
        c1 = -1 if l1 > 0 else 1
        c2 = -1 if l2 > 0 else 1
        c3 = -1 if l3 > 0 else 1
        h[v1] += 3*c1; h[v2] += 3*c2; h[v3] += 2*c3
        Jval = -c1*c2
        key = (min(v1,v2), max(v1,v2))
        J_dict[key] = J_dict.get(key, 0) + Jval
    return h, J_dict

def ising_energy(h, J_dict, s):
    E = h @ s
    for (i,j), Jij in J_dict.items():
        E += Jij * s[i] * s[j]
    return E

def count_sat(clauses, s):
    count = 0
    for clause in clauses:
        for lit in clause:
            if (lit > 0 and s[abs(lit)-1] > 0) or (lit < 0 and s[abs(lit)-1] < 0):
                count += 1; break
    return count

# ============================================================
# Init strategies (baseline)
# ============================================================
def init_random(N, rng):
    return rng.choice([-1.0, 1.0], size=N)

def init_bias_only(h):
    s = -np.sign(h)
    s[s == 0] = 1.0
    return s

def init_sequential_greedy(h, J_dict, N):
    s = np.zeros(N)
    h_eff = h.copy()
    fixed = np.zeros(N, dtype=bool)
    adj = [[] for _ in range(N)]
    for (i,j), Jij in J_dict.items():
        adj[i].append((j, Jij))
        adj[j].append((i, Jij))
    for _ in range(N):
        candidates = np.where(~fixed)[0]
        idx = candidates[np.argmax(np.abs(h_eff[candidates]))]
        s[idx] = -1.0 if h_eff[idx] > 0 else 1.0
        fixed[idx] = True
        for (j, Jij) in adj[idx]:
            if not fixed[j]:
                h_eff[j] += Jij * s[idx]
    return s

# ============================================================
# Optimised BP (Warning Propagation) for 3-SAT
# ============================================================
def build_factor_graph(n_vars, clauses):
    """Pre-compute factor graph structures for fast BP/SP."""
    m = len(clauses)
    # clause_arr[a] = (v0, v1, v2) variable indices
    # sign_arr[a] = (s0, s1, s2) signs (+1 or -1)
    clause_arr = np.zeros((m, 3), dtype=np.int32)
    sign_arr = np.zeros((m, 3), dtype=np.int8)

    # var_to_clauses_pos[i] = list of (clause_idx, pos_in_clause) where var i appears positive
    # var_to_clauses_neg[i] = list of (clause_idx, pos_in_clause) where var i appears negative
    vp = [[] for _ in range(n_vars)]
    vn = [[] for _ in range(n_vars)]

    for a, clause in enumerate(clauses):
        for k, lit in enumerate(clause):
            vi = abs(lit) - 1
            si = 1 if lit > 0 else -1
            clause_arr[a, k] = vi
            sign_arr[a, k] = si
            if si > 0:
                vp[vi].append((a, k))
            else:
                vn[vi].append((a, k))

    return clause_arr, sign_arr, vp, vn

def init_bp(n_vars, clauses, max_iter=200, damping=0.3, tol=1e-6):
    """
    Optimised BP for 3-SAT.

    Messages: eta[a,k] = warning from clause a to its k-th variable.
    This is the probability that clause a would be unsatisfied
    if variable k doesn't help.

    Update: eta[a,k] = prod_{j in clause a, j!=k} pi_{j→a}
    where pi_{j→a} = P(var j takes the "wrong" value for clause a | cavity)

    For variable j with sign s_j in clause a:
      Compute W+ = prod_{b in V+(j) \ a} (1-eta[b,pos(j,b)])
      Compute W- = prod_{b in V-(j) \ a} (1-eta[b,pos(j,b)])
      (where V+(j) = clauses wanting j=1, V-(j) = wanting j=0)

    If s_j = +1 (clause wants j=1):
      pi_{j→a} = P(j=0 | cavity)
        W+_cav = W+ / (1-eta[a,k])  [remove clause a from V+(j)]
        pi = W+_cav / (W+_cav + W-)
        = 1/(1 + W-/W+_cav)
        = 1/(1 + W-*(1-eta[a,k])/W+)

    If s_j = -1 (clause wants j=0):
      pi_{j→a} = P(j=1 | cavity)
        W-_cav = W- / (1-eta[a,k])
        pi = W-_cav / (W+ + W-_cav)
    """
    m = len(clauses)
    N = n_vars
    clause_arr, sign_arr, vp, vn = build_factor_graph(n_vars, clauses)

    # eta[a, k] messages
    eta = np.full((m, 3), 0.5)

    # Precompute log(1-eta) sums per variable for V+ and V-
    # log_W_plus[i] = sum_{(a,k) in vp[i]} log(1 - eta[a,k])
    # log_W_minus[i] = sum_{(a,k) in vn[i]} log(1 - eta[a,k])

    for iteration in range(max_iter):
        # Compute log W+ and W- for each variable
        log_W_plus = np.zeros(N)
        log_W_minus = np.zeros(N)

        log_one_minus_eta = np.log(np.maximum(1.0 - eta, 1e-15))

        for i in range(N):
            for (a, k) in vp[i]:
                log_W_plus[i] += log_one_minus_eta[a, k]
            for (a, k) in vn[i]:
                log_W_minus[i] += log_one_minus_eta[a, k]

        # Update eta
        new_eta = np.zeros((m, 3))

        for a in range(m):
            for k in range(3):
                prod = 1.0
                for kk in range(3):
                    if kk == k:
                        continue
                    j = clause_arr[a, kk]
                    sj = sign_arr[a, kk]
                    log_corr = log_one_minus_eta[a, kk]

                    if sj > 0:
                        # clause wants j=1, "wrong"=0
                        # pi = 1/(1 + exp(log_W_minus[j] - (log_W_plus[j] - log_corr)))
                        log_num = log_W_plus[j] - log_corr  # W+_cav
                        log_den_other = log_W_minus[j]       # W-
                    else:
                        # clause wants j=0, "wrong"=1
                        log_num = log_W_minus[j] - log_corr  # W-_cav
                        log_den_other = log_W_plus[j]         # W+

                    diff = log_den_other - log_num
                    if diff > 30:
                        pi_val = 0.0
                    elif diff < -30:
                        pi_val = 1.0
                    else:
                        pi_val = 1.0 / (1.0 + np.exp(diff))
                    prod *= pi_val

                new_eta[a, k] = prod

        # Damped update
        delta = np.max(np.abs(new_eta - eta))
        eta = damping * eta + (1 - damping) * new_eta

        if delta < tol:
            break

    # Compute marginals
    log_one_minus_eta = np.log(np.maximum(1.0 - eta, 1e-15))
    log_W_plus = np.zeros(N)
    log_W_minus = np.zeros(N)
    for i in range(N):
        for (a, k) in vp[i]:
            log_W_plus[i] += log_one_minus_eta[a, k]
        for (a, k) in vn[i]:
            log_W_minus[i] += log_one_minus_eta[a, k]

    # P(x_i=1) ∝ exp(-log_W_minus) ∝ 1/W_minus  -> high when W_minus small (many neg clauses need i)
    # Wait, let me reconsider:
    # P(x_i=0 is safe) ∝ W_plus = exp(log_W_plus) : no positive clause needs i
    # P(x_i=1 is safe) ∝ W_minus = exp(log_W_minus) : no negative clause needs i
    # So P(x_i=1) ∝ W_minus?  No — if W_plus is large, x_i=0 is safe → pressure AWAY from x_i=1.
    # Actually: P(x_i=1) ∝ prod of P(clause b ok if x_i=1)
    #   For positive clauses: x_i=1 satisfies them → always ok → factor 1
    #   For negative clauses: x_i=1 fails to satisfy them → ok only if other vars handle it
    #     → factor (1 - eta_{b→i})
    # So P(x_i=1) ∝ W_minus (product over negative clauses)
    # And P(x_i=0) ∝ W_plus (product over positive clauses)
    #
    # WRONG — it's the other way. W_plus = prod(1-eta) over POSITIVE clauses.
    # If eta is high for positive clauses, W_plus is small → positive clauses NEED i to be 1.
    # So small W_plus → pressure to set x_i=1.
    #
    # Correct assignment:
    # P(x_i=1) ∝ 1 / W_plus?  No...
    #
    # Let me think more carefully.
    # eta_{a→i} = P(clause a needs i). If i appears positive in a:
    #   clause a needs x_i=1. So high eta → pressure on x_i=1.
    # (1 - eta_{a→i}) = P(clause a does NOT need i).
    # W_plus = prod of (1 - eta) over positive clauses = P(NO positive clause needs i)
    # Small W_plus → positive clauses need i=1 → set x_i=1
    #
    # Similarly:
    # W_minus = prod of (1-eta) over negative clauses = P(NO negative clause needs i)
    # Small W_minus → negative clauses need i=0 → set x_i=0
    #
    # So: if W_plus < W_minus: positive clauses more needy → x_i=1 (s_i=+1)
    #     if W_minus < W_plus: negative clauses more needy → x_i=0 (s_i=-1)
    # i.e., log_W_plus < log_W_minus → s_i = +1

    s = np.ones(N)
    for i in range(N):
        if log_W_plus[i] < log_W_minus[i]:
            s[i] = 1.0   # positive clauses need i more → x_i = 1
        else:
            s[i] = -1.0  # negative clauses need i more → x_i = 0

    return s

# ============================================================
# Optimised Survey Propagation for 3-SAT
# ============================================================
def init_sp(n_vars, clauses, max_iter=200, damping=0.3, tol=1e-6, rng=None):
    """
    Survey Propagation on 3-SAT factor graph — optimised.

    For clause a, variable j (position kk), with sign s_j:
    We need sigma_u, sigma_s, sigma_0 for j→a.

    V_same(j) = clauses with same preference as a for j (excl. a itself)
    V_opp(j) = clauses with opposite preference

    pi_same = prod_{b in V_same(j)\a} (1 - eta[b, pos(j,b)])
    pi_opp  = prod_{b in V_opp(j)} (1 - eta[b, pos(j,b)])

    sigma_u = pi_same * (1 - pi_opp)
    sigma_s = pi_opp * (1 - pi_same)
    sigma_0 = pi_same * pi_opp

    factor = sigma_u / (sigma_u + sigma_s + sigma_0)

    eta[a,k] = prod_{kk != k} factor(kk)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    m = len(clauses)
    N = n_vars
    clause_arr, sign_arr, vp, vn = build_factor_graph(n_vars, clauses)

    eta = rng.uniform(0.0, 0.1, size=(m, 3))

    for iteration in range(max_iter):
        # Precompute log(1-eta)
        log_1m = np.log(np.maximum(1.0 - eta, 1e-15))

        # For each variable, compute total log(1-eta) sums for V+ and V-
        log_sum_plus = np.zeros(N)
        log_sum_minus = np.zeros(N)
        for i in range(N):
            for (a, k) in vp[i]:
                log_sum_plus[i] += log_1m[a, k]
            for (a, k) in vn[i]:
                log_sum_minus[i] += log_1m[a, k]

        new_eta = np.zeros((m, 3))

        for a in range(m):
            for k in range(3):
                prod = 1.0
                for kk in range(3):
                    if kk == k:
                        continue
                    j = clause_arr[a, kk]
                    sj = sign_arr[a, kk]
                    log_corr = log_1m[a, kk]

                    if sj > 0:
                        # same direction = V+(j), opposite = V-(j)
                        log_pi_same = log_sum_plus[j] - log_corr
                        log_pi_opp = log_sum_minus[j]
                    else:
                        log_pi_same = log_sum_minus[j] - log_corr
                        log_pi_opp = log_sum_plus[j]

                    pi_same = np.exp(np.clip(log_pi_same, -30, 0))
                    pi_opp = np.exp(np.clip(log_pi_opp, -30, 0))

                    sigma_u = pi_same * (1.0 - pi_opp)
                    sigma_s = pi_opp * (1.0 - pi_same)
                    sigma_0 = pi_same * pi_opp

                    denom = sigma_u + sigma_s + sigma_0
                    if denom < 1e-15:
                        factor = 0.5
                    else:
                        factor = sigma_u / denom

                    prod *= factor

                new_eta[a, k] = prod

        delta = np.max(np.abs(new_eta - eta))
        eta = damping * eta + (1 - damping) * new_eta

        if delta < tol:
            break

    # Assign based on total bias
    log_1m = np.log(np.maximum(1.0 - eta, 1e-15))
    s = np.ones(N)
    for i in range(N):
        pi_plus = 0.0
        for (a, k) in vp[i]:
            pi_plus += log_1m[a, k]
        pi_minus = 0.0
        for (a, k) in vn[i]:
            pi_minus += log_1m[a, k]

        W_plus = 1.0 - np.exp(np.clip(pi_plus, -30, 0))
        W_minus = 1.0 - np.exp(np.clip(pi_minus, -30, 0))

        if W_plus > W_minus:
            s[i] = 1.0
        elif W_minus > W_plus:
            s[i] = -1.0
        else:
            s[i] = 1.0

    return s

# ============================================================
# Find all benchmark sets
# ============================================================
sets = {}
for cnf_path in glob.glob(os.path.join(BENCH_ROOT, "**", "uf*.cnf"), recursive=True):
    fname = os.path.basename(cnf_path)
    parts = fname.split('-')
    prefix = parts[0]
    if prefix not in sets:
        sets[prefix] = []
    sets[prefix].append(cnf_path)

sorted_sets = sorted(sets.items(), key=lambda x: int(x[0][2:]))

rng = np.random.default_rng(42)
N_RANDOM = 50

print("="*100)
print("  SATLIB Initialisation Benchmark — De Morgan Encoding")
print("  Methods: Random(×50), Bias-only, Seq Greedy, BP, SP")
print("="*100)

all_results = []

for prefix, paths in sorted_sets:
    paths.sort()
    n_inst = len(paths)

    # Limit large sets to 100 for speed
    if n_inst > 100:
        paths = paths[:100]
        n_inst = 100

    n0, c0 = read_dimacs(paths[0])
    m0 = len(c0)
    alpha = m0 / n0

    print(f"\n{'─'*100}")
    print(f"  {prefix}: n={n0}, m={m0}, α={alpha:.3f}, {n_inst} instances")
    print(f"{'─'*100}")

    t0 = time.time()

    stats = {k: {'sat':[], 'unsat':[], 'energy':[]}
             for k in ['random','bias','seq_greedy','bp','sp']}

    for pi_idx, path in enumerate(paths):
        n_vars, clauses_list = read_dimacs(path)
        m = len(clauses_list)
        h, J_dict = demorgan_encode(n_vars, clauses_list)

        # Random (averaged)
        r_sats = []
        r_es = []
        for _ in range(N_RANDOM):
            s = init_random(n_vars, rng)
            r_sats.append(count_sat(clauses_list, s))
            r_es.append(ising_energy(h, J_dict, s))
        stats['random']['sat'].append(np.mean(r_sats))
        stats['random']['unsat'].append(m - np.mean(r_sats))
        stats['random']['energy'].append(np.mean(r_es))

        # Bias-only
        s = init_bias_only(h)
        sat_b = count_sat(clauses_list, s)
        stats['bias']['sat'].append(sat_b)
        stats['bias']['unsat'].append(m - sat_b)
        stats['bias']['energy'].append(ising_energy(h, J_dict, s))

        # Seq greedy
        s = init_sequential_greedy(h, J_dict, n_vars)
        sat_g = count_sat(clauses_list, s)
        stats['seq_greedy']['sat'].append(sat_g)
        stats['seq_greedy']['unsat'].append(m - sat_g)
        stats['seq_greedy']['energy'].append(ising_energy(h, J_dict, s))

        # BP
        s = init_bp(n_vars, clauses_list)
        sat_bp = count_sat(clauses_list, s)
        stats['bp']['sat'].append(sat_bp)
        stats['bp']['unsat'].append(m - sat_bp)
        stats['bp']['energy'].append(ising_energy(h, J_dict, s))

        # SP
        s = init_sp(n_vars, clauses_list, rng=rng)
        sat_sp = count_sat(clauses_list, s)
        stats['sp']['sat'].append(sat_sp)
        stats['sp']['unsat'].append(m - sat_sp)
        stats['sp']['energy'].append(ising_energy(h, J_dict, s))

        if (pi_idx+1) % 25 == 0 or pi_idx == n_inst - 1:
            elapsed_so_far = time.time() - t0
            print(f"    [{pi_idx+1}/{n_inst}] {elapsed_so_far:.1f}s", flush=True)

    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print(f"\n  {'Method':<18s}  {'SAT% (mean±std)':>18s}  {'UNSAT (mean±std)':>20s}  {'E (mean)':>12s}")
    print(f"  {'─'*75}")

    row = {'prefix': prefix, 'n': n0, 'm': m0, 'n_inst': n_inst}
    for method, label in [('random','Random (×50)'), ('bias','Bias-only'),
                          ('seq_greedy','Seq. greedy'), ('bp','BP'), ('sp','SP')]:
        sat_arr = np.array(stats[method]['sat'])
        unsat_arr = np.array(stats[method]['unsat'])
        e_arr = np.array(stats[method]['energy'])
        sat_pct = np.mean(sat_arr) / m0 * 100
        sat_std = np.std(sat_arr) / m0 * 100
        unsat_m = np.mean(unsat_arr)
        unsat_s = np.std(unsat_arr)
        e_m = np.mean(e_arr)
        print(f"  {label:<18s}  {sat_pct:>7.2f} ± {sat_std:>5.2f}%  {unsat_m:>8.1f} ± {unsat_s:>6.1f}    {e_m:>10.1f}")
        row[f'{method}_sat_pct'] = sat_pct
        row[f'{method}_sat_std'] = sat_std
        row[f'{method}_unsat'] = unsat_m
        row[f'{method}_unsat_std'] = unsat_s

    # Head-to-head
    b = np.array(stats['bias']['sat'])
    bp_arr = np.array(stats['bp']['sat'])
    sp_arr = np.array(stats['sp']['sat'])

    print(f"\n  Head-to-head vs Bias-only:")
    print(f"    BP:   Bias>{np.sum(b>bp_arr)}, Tie={np.sum(b==bp_arr)}, BP>{np.sum(b<bp_arr)}")
    print(f"    SP:   Bias>{np.sum(b>sp_arr)}, Tie={np.sum(b==sp_arr)}, SP>{np.sum(b<sp_arr)}")

    all_results.append(row)

# Final summary
print(f"\n\n{'='*110}")
print("  SUMMARY TABLE")
print(f"{'='*110}")
print(f"  {'Set':<10s} {'n':>4s} {'m':>5s} {'#':>4s}  "
      f"{'Rand%':>6s}  {'Bias%':>6s}  {'Seq%':>6s}  {'BP%':>6s}  {'SP%':>6s}  "
      f"{'Rand U':>7s}  {'Bias U':>7s}  {'Seq U':>7s}  {'BP U':>7s}  {'SP U':>7s}")
print(f"  {'─'*108}")
for r in all_results:
    print(f"  {r['prefix']:<10s} {r['n']:>4d} {r['m']:>5d} {r['n_inst']:>4d}  "
          f"{r['random_sat_pct']:>5.1f}%  {r['bias_sat_pct']:>5.1f}%  {r['seq_greedy_sat_pct']:>5.1f}%  "
          f"{r['bp_sat_pct']:>5.1f}%  {r['sp_sat_pct']:>5.1f}%  "
          f"{r['random_unsat']:>7.1f}  {r['bias_unsat']:>7.1f}  {r['seq_greedy_unsat']:>7.1f}  "
          f"{r['bp_unsat']:>7.1f}  {r['sp_unsat']:>7.1f}")
