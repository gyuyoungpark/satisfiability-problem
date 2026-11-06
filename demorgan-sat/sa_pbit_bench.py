#!/usr/bin/env python3
"""
SA and p-bit solvers with BP/Bias/Random initialisation.
Two modes:
  (1) Ising-SA / Ising-pbit: minimise Ising energy, but TRACK best SAT config
  (2) SAT-SA: minimise UNSAT count directly (models SAT-aware sampler)

Benchmark on SATLIB uf* instances with De Morgan encoding.
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
    J = np.zeros((n_vars, n_vars))
    for clause in clauses:
        l1, l2, l3 = clause[0], clause[1], clause[2]
        v1, v2, v3 = abs(l1)-1, abs(l2)-1, abs(l3)-1
        c1 = -1 if l1 > 0 else 1
        c2 = -1 if l2 > 0 else 1
        c3 = -1 if l3 > 0 else 1
        h[v1] += 3*c1; h[v2] += 3*c2; h[v3] += 2*c3
        J[v1, v2] += -c1*c2
        J[v2, v1] += -c1*c2
    return h, J

def count_sat(clauses, s):
    count = 0
    for clause in clauses:
        for lit in clause:
            if (lit > 0 and s[abs(lit)-1] > 0) or (lit < 0 and s[abs(lit)-1] < 0):
                count += 1; break
    return count

# ============================================================
# Precompute clause-variable structure for fast UNSAT counting
# ============================================================
def build_clause_var_map(n_vars, clauses):
    """For each variable, which clauses contain it and with what sign."""
    m = len(clauses)
    var_to_clauses = [[] for _ in range(n_vars)]  # (clause_idx, lit_position, sign)
    clause_lit_vars = []  # clause_lit_vars[a] = [(var, sign), ...]

    for a, clause in enumerate(clauses):
        lits = []
        for k, lit in enumerate(clause):
            vi = abs(lit) - 1
            si = 1 if lit > 0 else -1
            var_to_clauses[vi].append((a, k, si))
            lits.append((vi, si))
        clause_lit_vars.append(lits)

    return var_to_clauses, clause_lit_vars

def compute_clause_sat(clauses, clause_lit_vars, s):
    """For each clause, how many literals are satisfied."""
    m = len(clauses)
    clause_nsat = np.zeros(m, dtype=np.int32)
    for a, lits in enumerate(clause_lit_vars):
        for (vi, si) in lits:
            if s[vi] * si > 0:
                clause_nsat[a] += 1
    return clause_nsat

# ============================================================
# Initialisation
# ============================================================
def init_random(N, rng):
    return rng.choice([-1.0, 1.0], size=N)

def init_bias_only(h):
    s = -np.sign(h)
    s[s == 0] = 1.0
    return s

def build_factor_graph(n_vars, clauses):
    m = len(clauses)
    clause_arr = np.zeros((m, 3), dtype=np.int32)
    sign_arr = np.zeros((m, 3), dtype=np.int8)
    vp = [[] for _ in range(n_vars)]
    vn = [[] for _ in range(n_vars)]
    for a, clause in enumerate(clauses):
        for k, lit in enumerate(clause):
            vi = abs(lit) - 1
            si = 1 if lit > 0 else -1
            clause_arr[a, k] = vi
            sign_arr[a, k] = si
            (vp if si > 0 else vn)[vi].append((a, k))
    return clause_arr, sign_arr, vp, vn

def init_bp(n_vars, clauses, max_iter=200, damping=0.3, tol=1e-6):
    m = len(clauses); N = n_vars
    clause_arr, sign_arr, vp, vn = build_factor_graph(n_vars, clauses)
    eta = np.full((m, 3), 0.5)
    for iteration in range(max_iter):
        log_1m = np.log(np.maximum(1.0 - eta, 1e-15))
        lWp = np.zeros(N); lWn = np.zeros(N)
        for i in range(N):
            for (a, k) in vp[i]: lWp[i] += log_1m[a, k]
            for (a, k) in vn[i]: lWn[i] += log_1m[a, k]
        new_eta = np.zeros((m, 3))
        for a in range(m):
            for k in range(3):
                prod = 1.0
                for kk in range(3):
                    if kk == k: continue
                    j = clause_arr[a, kk]; sj = sign_arr[a, kk]; lc = log_1m[a, kk]
                    if sj > 0: ln = lWp[j]-lc; ld = lWn[j]
                    else: ln = lWn[j]-lc; ld = lWp[j]
                    d = ld - ln
                    if d > 30: pv = 0.0
                    elif d < -30: pv = 1.0
                    else: pv = 1.0/(1.0+np.exp(d))
                    prod *= pv
                new_eta[a, k] = prod
        delta = np.max(np.abs(new_eta - eta))
        eta = damping*eta + (1-damping)*new_eta
        if delta < tol: break
    log_1m = np.log(np.maximum(1.0 - eta, 1e-15))
    lWp = np.zeros(N); lWn = np.zeros(N)
    for i in range(N):
        for (a, k) in vp[i]: lWp[i] += log_1m[a, k]
        for (a, k) in vn[i]: lWn[i] += log_1m[a, k]
    s = np.ones(N)
    for i in range(N): s[i] = 1.0 if lWp[i] < lWn[i] else -1.0
    return s

# ============================================================
# Ising SA — track both best-energy and best-SAT configs
# ============================================================
def run_ising_sa(h, J_mat, s0, clauses, clause_lit_vars, var_to_clauses, rng,
                 T_start=5.0, T_end=0.01, n_sweeps=200):
    N = len(s0)
    s = s0.copy()
    m = len(clauses)

    h_eff = h + J_mat @ s
    E = np.dot(h, s) + 0.5 * s @ J_mat @ s

    # Compute clause satisfaction counts
    clause_nsat = compute_clause_sat(clauses, clause_lit_vars, s)
    n_unsat = int(np.sum(clause_nsat == 0))
    cur_sat = m - n_unsat

    best_sat = cur_sat
    best_sat_s = s.copy()

    temps = np.linspace(T_start, T_end, n_sweeps)

    for sweep, T in enumerate(temps):
        order = rng.permutation(N)
        for i in order:
            dE = -2.0 * s[i] * h_eff[i]
            if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
                old_si = s[i]
                s[i] = -s[i]
                E += dE
                h_eff += J_mat[:, i] * 2.0 * s[i]

                # Update clause_nsat for affected clauses
                for (a, k, sign) in var_to_clauses[i]:
                    if old_si * sign > 0:
                        # Was satisfying, now not
                        old_nsat = clause_nsat[a]
                        clause_nsat[a] -= 1
                        if clause_nsat[a] == 0 and old_nsat > 0:
                            n_unsat += 1
                    else:
                        # Was not satisfying, now is
                        old_nsat = clause_nsat[a]
                        clause_nsat[a] += 1
                        if old_nsat == 0:
                            n_unsat -= 1

                cur_sat = m - n_unsat
                if cur_sat > best_sat:
                    best_sat = cur_sat
                    best_sat_s = s.copy()
                    if best_sat == m:
                        return best_sat_s, best_sat

    return best_sat_s, best_sat

# ============================================================
# Ising p-bit — track best-SAT config
# ============================================================
def run_ising_pbit(h, J_mat, s0, clauses, clause_lit_vars, var_to_clauses, rng,
                   T_start=5.0, T_end=0.01, n_sweeps=200):
    N = len(s0)
    s = s0.copy()
    m = len(clauses)

    h_eff = h + J_mat @ s
    clause_nsat = compute_clause_sat(clauses, clause_lit_vars, s)
    n_unsat = int(np.sum(clause_nsat == 0))
    cur_sat = m - n_unsat

    best_sat = cur_sat
    best_sat_s = s.copy()

    temps = np.linspace(T_start, T_end, n_sweeps)

    for sweep, T in enumerate(temps):
        order = rng.permutation(N)
        for i in order:
            arg = -2.0 * h_eff[i] / max(T, 1e-10)
            if arg > 30: p_plus = 1.0
            elif arg < -30: p_plus = 0.0
            else: p_plus = 1.0 / (1.0 + np.exp(-arg))
            new_si = 1.0 if rng.random() < p_plus else -1.0

            if new_si != s[i]:
                old_si = s[i]
                dE = -2.0 * s[i] * h_eff[i]
                s[i] = new_si
                h_eff += J_mat[:, i] * 2.0 * s[i]

                for (a, k, sign) in var_to_clauses[i]:
                    if old_si * sign > 0:
                        old_nsat = clause_nsat[a]
                        clause_nsat[a] -= 1
                        if clause_nsat[a] == 0 and old_nsat > 0:
                            n_unsat += 1
                    else:
                        old_nsat = clause_nsat[a]
                        clause_nsat[a] += 1
                        if old_nsat == 0:
                            n_unsat -= 1

                cur_sat = m - n_unsat
                if cur_sat > best_sat:
                    best_sat = cur_sat
                    best_sat_s = s.copy()
                    if best_sat == m:
                        return best_sat_s, best_sat

    return best_sat_s, best_sat

# ============================================================
# SAT-count SA (WalkSAT-style Metropolis on UNSAT count)
# ============================================================
def run_sat_sa(clauses, clause_lit_vars, var_to_clauses, s0, n_vars, rng,
               T_start=2.0, T_end=0.001, n_sweeps=200):
    """SA where energy = number of unsatisfied clauses."""
    N = n_vars
    m = len(clauses)
    s = s0.copy()

    clause_nsat = compute_clause_sat(clauses, clause_lit_vars, s)
    n_unsat = int(np.sum(clause_nsat == 0))

    best_unsat = n_unsat
    best_s = s.copy()

    temps = np.linspace(T_start, T_end, n_sweeps)

    for sweep, T in enumerate(temps):
        order = rng.permutation(N)
        for i in order:
            # Compute delta_unsat if we flip spin i
            delta_unsat = 0
            for (a, k, sign) in var_to_clauses[i]:
                if s[i] * sign > 0:
                    # Currently satisfying this literal
                    if clause_nsat[a] == 1:
                        # Only satisfying literal → clause becomes unsat
                        delta_unsat += 1
                else:
                    # Currently not satisfying
                    if clause_nsat[a] == 0:
                        # Clause currently unsat → becomes sat
                        delta_unsat -= 1

            if delta_unsat <= 0 or rng.random() < np.exp(-delta_unsat / max(T, 1e-10)):
                old_si = s[i]
                s[i] = -s[i]
                n_unsat += delta_unsat

                for (a, k, sign) in var_to_clauses[i]:
                    if old_si * sign > 0:
                        clause_nsat[a] -= 1
                    else:
                        clause_nsat[a] += 1

                if n_unsat < best_unsat:
                    best_unsat = n_unsat
                    best_s = s.copy()
                    if best_unsat == 0:
                        return best_s, m

    return best_s, m - best_unsat

# ============================================================
# Main benchmark
# ============================================================
if __name__ == "__main__":
    sets = {}
    for cnf_path in glob.glob(os.path.join(BENCH_ROOT, "**", "uf*.cnf"), recursive=True):
        fname = os.path.basename(cnf_path)
        prefix = fname.split('-')[0]
        if prefix not in sets:
            sets[prefix] = []
        sets[prefix].append(cnf_path)

    sorted_sets = sorted(sets.items(), key=lambda x: int(x[0][2:]))
    rng = np.random.default_rng(42)

    N_SWEEPS = 200

    print("=" * 120)
    print("  SA / p-bit / SAT-SA Solver Benchmark — De Morgan Encoding")
    print(f"  Sweeps={N_SWEEPS}")
    print("  Init: Random, Bias-only, BP")
    print("  Solvers: Ising-SA, Ising-pbit, SAT-SA (UNSAT-count based)")
    print("=" * 120)

    all_results = []

    for prefix, paths in sorted_sets:
        paths.sort()
        n_inst = min(len(paths), 100)
        paths = paths[:n_inst]

        n0, c0 = read_dimacs(paths[0])
        m0 = len(c0)

        print(f"\n{'─'*120}")
        print(f"  {prefix}: n={n0}, m={m0}, {n_inst} instances")
        print(f"{'─'*120}")

        t0 = time.time()

        # 3 inits × 3 solvers = 9 combos
        configs = []
        for init_name in ['random', 'bias', 'bp']:
            for solver_name in ['ising_sa', 'ising_pbit', 'sat_sa']:
                configs.append((init_name, solver_name))

        stats = {f'{i}_{s}': {'sat': [], 'solved': 0} for i, s in configs}

        for pi_idx, path in enumerate(paths):
            n_vars, clauses_list = read_dimacs(path)
            m = len(clauses_list)
            h, J_mat = demorgan_encode(n_vars, clauses_list)
            v2c, clv = build_clause_var_map(n_vars, clauses_list)

            s_rand = init_random(n_vars, rng)
            s_bias = init_bias_only(h)
            s_bp_v = init_bp(n_vars, clauses_list)
            inits = {'random': s_rand, 'bias': s_bias, 'bp': s_bp_v}

            for init_name, s0 in inits.items():
                # Ising SA
                best_s, best_sat = run_ising_sa(h, J_mat, s0.copy(), clauses_list, clv, v2c, rng,
                                                5.0, 0.01, N_SWEEPS)
                key = f'{init_name}_ising_sa'
                stats[key]['sat'].append(best_sat)
                if best_sat == m: stats[key]['solved'] += 1

                # Ising p-bit
                best_s, best_sat = run_ising_pbit(h, J_mat, s0.copy(), clauses_list, clv, v2c, rng,
                                                  5.0, 0.01, N_SWEEPS)
                key = f'{init_name}_ising_pbit'
                stats[key]['sat'].append(best_sat)
                if best_sat == m: stats[key]['solved'] += 1

                # SAT-SA
                best_s, best_sat = run_sat_sa(clauses_list, clv, v2c, s0.copy(), n_vars, rng,
                                              2.0, 0.001, N_SWEEPS)
                key = f'{init_name}_sat_sa'
                stats[key]['sat'].append(best_sat)
                if best_sat == m: stats[key]['solved'] += 1

            if (pi_idx+1) % 10 == 0 or pi_idx == n_inst - 1:
                elapsed = time.time() - t0
                print(f"    [{pi_idx+1}/{n_inst}] {elapsed:.1f}s", flush=True)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s\n")

        print(f"  {'Init':<8s} {'Solver':<12s} {'SAT%':>8s} {'Solved':>12s} {'UNSAT':>8s}")
        print(f"  {'─'*55}")

        row = {'prefix': prefix, 'n': n0, 'm': m0, 'n_inst': n_inst}
        for init_name in ['random', 'bias', 'bp']:
            for solver_name in ['ising_sa', 'ising_pbit', 'sat_sa']:
                key = f'{init_name}_{solver_name}'
                sa = np.array(stats[key]['sat'])
                pct = np.mean(sa) / m0 * 100
                solved = stats[key]['solved']
                unsat = np.mean(m0 - sa)
                ilabel = {'random':'Rand', 'bias':'Bias', 'bp':'BP'}[init_name]
                slabel = {'ising_sa':'Ising-SA', 'ising_pbit':'Ising-pb', 'sat_sa':'SAT-SA'}[solver_name]
                marker = ' ***' if solved == n_inst else ''
                print(f"  {ilabel:<8s} {slabel:<12s} {pct:>7.2f}%  {solved:>4d}/{n_inst:<4d}  {unsat:>6.2f}{marker}")
                row[f'{key}_pct'] = pct
                row[f'{key}_solved'] = solved
                row[f'{key}_unsat'] = unsat

        all_results.append(row)

    # Final summary
    print(f"\n\n{'='*130}")
    print("  SUMMARY: Solved instances (all clauses satisfied)")
    print(f"{'='*130}")
    print(f"  {'Set':<8s} {'n':>4s}  "
          f"{'R+ISA':>6s} {'B+ISA':>6s} {'BP+ISA':>7s}  "
          f"{'R+Ipb':>6s} {'B+Ipb':>6s} {'BP+Ipb':>7s}  "
          f"{'R+SSA':>6s} {'B+SSA':>6s} {'BP+SSA':>7s}")
    print(f"  {'─'*85}")
    for r in all_results:
        print(f"  {r['prefix']:<8s} {r['n']:>4d}  "
              f"{r['random_ising_sa_solved']:>6d} {r['bias_ising_sa_solved']:>6d} {r['bp_ising_sa_solved']:>7d}  "
              f"{r['random_ising_pbit_solved']:>6d} {r['bias_ising_pbit_solved']:>6d} {r['bp_ising_pbit_solved']:>7d}  "
              f"{r['random_sat_sa_solved']:>6d} {r['bias_sat_sa_solved']:>6d} {r['bp_sat_sa_solved']:>7d}")
