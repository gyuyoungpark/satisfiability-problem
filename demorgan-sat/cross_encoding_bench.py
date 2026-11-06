#!/usr/bin/env python3
"""
Cross-encoding benchmark v2: adaptive temperature per encoding.
De Morgan, Nüßlein, Chancellor, OR-chain on SATLIB.
SA and p-bit with random init, 200 sweeps.
T_start set to max(|h_eff|) so all encodings explore comparably.
"""
import numpy as np
import os, glob, time

from benchmark_paths import benchmark_root

BENCH_ROOT = benchmark_root()

def read_dimacs(path):
    clauses = []; n_vars = 0
    with open(path) as f:
        buf = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('c') or line.startswith('%'): continue
            if line.startswith('p'): n_vars = int(line.split()[2]); continue
            for v in map(int, line.split()):
                if v == 0:
                    if buf: clauses.append(buf); buf = []
                else: buf.append(v)
    return n_vars, clauses

def count_sat(clauses, s):
    c = 0
    for clause in clauses:
        for lit in clause:
            if (lit > 0 and s[abs(lit)-1] > 0) or (lit < 0 and s[abs(lit)-1] < 0):
                c += 1; break
    return c

# ---- Encodings ----
def encode_demorgan(nv, cl):
    N = nv; h = np.zeros(N); J = np.zeros((N, N))
    for clause in cl:
        l1,l2,l3 = clause; v1,v2,v3 = abs(l1)-1,abs(l2)-1,abs(l3)-1
        c1 = -1 if l1>0 else 1; c2 = -1 if l2>0 else 1; c3 = -1 if l3>0 else 1
        h[v1]+=3*c1; h[v2]+=3*c2; h[v3]+=2*c3
        J[v1,v2]+=-c1*c2; J[v2,v1]+=-c1*c2
    return N, h, J

def get_nusslein_qubo(sgns):
    Q = np.zeros((4,4))
    if sgns==(1,1,1): Q[0,1]=2;Q[0,3]=-2;Q[1,3]=-2;Q[2,2]=-1;Q[2,3]=1;Q[3,3]=1
    elif sgns==(1,1,-1): Q[0,1]=2;Q[0,3]=-2;Q[1,3]=-2;Q[2,2]=1;Q[2,3]=-1;Q[3,3]=2
    elif sgns==(1,-1,-1): Q[0,0]=2;Q[0,1]=-2;Q[0,3]=-2;Q[1,3]=2;Q[2,2]=1;Q[2,3]=-1
    else: Q[0,0]=-1;Q[0,1]=1;Q[0,2]=1;Q[0,3]=1;Q[1,1]=-1;Q[1,2]=1;Q[1,3]=1;Q[2,2]=-1;Q[2,3]=1;Q[3,3]=-1
    return Q

def encode_nusslein(nv, cl):
    m=len(cl); N=nv+m; h=np.zeros(N); J=np.zeros((N,N))
    for k,clause in enumerate(cl):
        lits=list(clause); order=sorted(range(3),key=lambda i:lits[i]<0); lits=[lits[i] for i in order]
        sgns=tuple(1 if l>0 else -1 for l in lits); vis=[abs(l)-1 for l in lits]; ai=nv+k
        Q=get_nusslein_qubo(sgns); idx=vis+[ai]
        for a in range(4):
            if Q[a,a]!=0: h[idx[a]]+=Q[a,a]/2
            for b in range(a+1,4):
                if Q[a,b]!=0:
                    ga,gb=idx[a],idx[b]; J[ga,gb]+=Q[a,b]/4; J[gb,ga]+=Q[a,b]/4; h[ga]+=Q[a,b]/4; h[gb]+=Q[a,b]/4
    return N, h, J

def encode_chancellor(nv, cl):
    m=len(cl); N=nv+m; h=np.zeros(N); J=np.zeros((N,N))
    qubo_map = {
        (1,1,1): [(-8,0,0),(4,0,1),(4,0,2),(4,0,3),(-8,1,1),(4,1,2),(4,1,3),(-8,2,2),(4,2,3),(-8,3,3)],
        (1,1,-1): [(-4,0,0),(4,0,1),(-4,0,2),(4,0,3),(-4,1,1),(-4,1,2),(4,1,3),(-4,2,2),(-4,2,3),(-8,3,3)],
        (1,-1,-1): [(-4,0,0),(-4,0,1),(-4,0,2),(4,0,3),(-4,1,1),(4,1,2),(-4,1,3),(-4,2,2),(-4,2,3),(-4,3,3)],
        (-1,-1,-1): [(-4,0,0),(4,0,1),(4,0,2),(-4,0,3),(-4,1,1),(4,1,2),(-4,1,3),(-4,2,2),(-4,2,3),(-4,3,3)],
    }
    for k,clause in enumerate(cl):
        lits=list(clause); order=sorted(range(3),key=lambda i:lits[i]<0); lits=[lits[i] for i in order]
        sgns=tuple(1 if l>0 else -1 for l in lits); vis=[abs(l)-1 for l in lits]; ai=nv+k; idx=vis+[ai]
        for (val,a,b) in qubo_map[sgns]:
            if a==b:
                h[idx[a]]+=val/2
            else:
                ga,gb=idx[a],idx[b]; J[ga,gb]+=val/4; J[gb,ga]+=val/4; h[ga]+=val/4; h[gb]+=val/4
    return N, h, J

def encode_orchain(nv, cl):
    m=len(cl); N=nv+m; h=np.zeros(N); J=np.zeros((N,N))
    hT=[1.0,1.0,-1.0,-3.0]; edges=[(0,1,1.0),(0,3,-2.0),(1,3,-2.0),(2,3,1.0)]
    for k,clause in enumerate(cl):
        l1,l2,l3=clause; v1,v2,v3=abs(l1)-1,abs(l2)-1,abs(l3)-1
        s1=1 if l1>0 else -1; s2=1 if l2>0 else -1; s3=1 if l3>0 else -1
        idx=[v1,v2,v3,nv+k]; sgn=[s1,s2,s3,1]
        for p in range(4): h[idx[p]]+=hT[p]*sgn[p]
        for (a,b,Jv) in edges:
            c=Jv*sgn[a]*sgn[b]; ga,gb=idx[a],idx[b]; J[ga,gb]+=c; J[gb,ga]+=c
    return N, h, J

# ---- Clause tracking ----
def build_clause_var_map(nv, cl):
    v2c=[[] for _ in range(nv)]; clv=[]
    for a,clause in enumerate(cl):
        lits=[]
        for k,lit in enumerate(clause):
            vi=abs(lit)-1; si=1 if lit>0 else -1; v2c[vi].append((a,k,si)); lits.append((vi,si))
        clv.append(lits)
    return v2c, clv

def compute_clause_nsat(cl, clv, s):
    m=len(cl); cn=np.zeros(m,dtype=np.int32)
    for a,lits in enumerate(clv):
        for vi,si in lits:
            if s[vi]*si>0: cn[a]+=1
    return cn

# ---- Solvers ----
def run_sa(h, J_mat, s0, n_orig, cl, clv, v2c, rng, T_start, T_end, n_sweeps):
    N=len(s0); s=s0.copy(); m=len(cl)
    h_eff=h+J_mat@s; cn=compute_clause_nsat(cl,clv,s[:n_orig]); nu=int(np.sum(cn==0))
    best_sat=m-nu; best_s=s[:n_orig].copy()
    temps=np.linspace(T_start,T_end,n_sweeps)
    for T in temps:
        for i in rng.permutation(N):
            dE=-2.0*s[i]*h_eff[i]
            if dE<0 or rng.random()<np.exp(-dE/max(T,1e-10)):
                old=s[i]; s[i]=-s[i]; h_eff+=J_mat[:,i]*2.0*s[i]
                if i<n_orig:
                    for (a,k,sign) in v2c[i]:
                        if old*sign>0:
                            on=cn[a]; cn[a]-=1
                            if cn[a]==0 and on>0: nu+=1
                        else:
                            on=cn[a]; cn[a]+=1
                            if on==0: nu-=1
                    cs=m-nu
                    if cs>best_sat: best_sat=cs; best_s=s[:n_orig].copy()
                    if best_sat==m: return best_s,best_sat
    return best_s,best_sat

def run_pbit(h, J_mat, s0, n_orig, cl, clv, v2c, rng, T_start, T_end, n_sweeps):
    N=len(s0); s=s0.copy(); m=len(cl)
    h_eff=h+J_mat@s; cn=compute_clause_nsat(cl,clv,s[:n_orig]); nu=int(np.sum(cn==0))
    best_sat=m-nu; best_s=s[:n_orig].copy()
    temps=np.linspace(T_start,T_end,n_sweeps)
    for T in temps:
        for i in rng.permutation(N):
            arg=-2.0*h_eff[i]/max(T,1e-10)
            if arg>30: pp=1.0
            elif arg<-30: pp=0.0
            else: pp=1.0/(1.0+np.exp(-arg))
            ns=1.0 if rng.random()<pp else -1.0
            if ns!=s[i]:
                old=s[i]; s[i]=ns; h_eff+=J_mat[:,i]*2.0*s[i]
                if i<n_orig:
                    for (a,k,sign) in v2c[i]:
                        if old*sign>0:
                            on=cn[a]; cn[a]-=1
                            if cn[a]==0 and on>0: nu+=1
                        else:
                            on=cn[a]; cn[a]+=1
                            if on==0: nu-=1
                    cs=m-nu
                    if cs>best_sat: best_sat=cs; best_s=s[:n_orig].copy()
                    if best_sat==m: return best_s,best_sat
    return best_s,best_sat

# ============================================================
if __name__ == "__main__":
    sets = {}
    for cnf_path in glob.glob(os.path.join(BENCH_ROOT, "**", "uf*.cnf"), recursive=True):
        fname = os.path.basename(cnf_path)
        prefix = fname.split('-')[0]
        if prefix not in sets: sets[prefix] = []
        sets[prefix].append(cnf_path)

    sorted_sets = sorted(sets.items(), key=lambda x: int(x[0][2:]))
    rng = np.random.default_rng(42)
    N_SWEEPS = 200

    encoders = {'DeMorgan': encode_demorgan, 'Nusslein': encode_nusslein,
                'Chancellor': encode_chancellor, 'ORchain': encode_orchain}

    print("="*110)
    print("  Cross-Encoding Benchmark (adaptive T_start)")
    print("="*110)

    all_results = []

    for prefix, paths in sorted_sets:
        paths.sort()
        n_inst = min(len(paths), 100)
        paths = paths[:n_inst]
        n0, c0 = read_dimacs(paths[0]); m0 = len(c0)

        print(f"\n{'─'*110}")
        print(f"  {prefix}: n={n0}, m={m0}, {n_inst} instances")
        print(f"{'─'*110}")

        t0 = time.time()
        stats = {}
        for enc in encoders:
            for sol in ['sa','pbit']:
                stats[(enc,sol)] = {'sat':[],'solved':0}

        for pi, path in enumerate(paths):
            nv, cl = read_dimacs(path); m = len(cl)
            v2c, clv = build_clause_var_map(nv, cl)

            for enc_name, enc_fn in encoders.items():
                N, h, J = enc_fn(nv, cl)
                # Adaptive T_start: scale to energy landscape
                h_scale = np.max(np.abs(h)) if len(h)>0 else 1.0
                J_scale = np.max(np.abs(J)) if J.size>0 else 0.0
                T_start = max(2.0 * (h_scale + J_scale * N**0.5), 5.0)
                T_end = 0.01

                s0 = rng.choice([-1.0,1.0], size=N)

                bs,bsat = run_sa(h, J, s0.copy(), nv, cl, clv, v2c, rng, T_start, T_end, N_SWEEPS)
                stats[(enc_name,'sa')]['sat'].append(bsat)
                if bsat==m: stats[(enc_name,'sa')]['solved']+=1

                bs,bsat = run_pbit(h, J, s0.copy(), nv, cl, clv, v2c, rng, T_start, T_end, N_SWEEPS)
                stats[(enc_name,'pbit')]['sat'].append(bsat)
                if bsat==m: stats[(enc_name,'pbit')]['solved']+=1

            if (pi+1)%10==0 or pi==n_inst-1:
                print(f"    [{pi+1}/{n_inst}] {time.time()-t0:.1f}s", flush=True)

        elapsed = time.time()-t0
        print(f"  Done in {elapsed:.1f}s\n")
        print(f"  {'Encoding':<12s} {'Solver':<6s} {'N':>5s} {'SAT%':>8s} {'Solved':>10s} {'UNSAT':>8s}")
        print(f"  {'─'*55}")

        row = {'prefix':prefix,'n':n0,'m':m0,'n_inst':n_inst}
        for enc in encoders:
            Nt = n0 if enc=='DeMorgan' else n0+m0
            for sol in ['sa','pbit']:
                st=stats[(enc,sol)]; sa=np.array(st['sat'])
                pct=np.mean(sa)/m0*100; solved=st['solved']; unsat=np.mean(m0-sa)
                sl='SA' if sol=='sa' else 'pb'
                print(f"  {enc:<12s} {sl:<6s} {Nt:>5d} {pct:>7.2f}%  {solved:>3d}/{n_inst:<3d}  {unsat:>6.2f}")
                row[f'{enc}_{sol}_pct']=pct; row[f'{enc}_{sol}_solved']=solved

        all_results.append(row)

    print(f"\n{'='*90}")
    print("  SUMMARY: Solved (SA) / Solved (p-bit)")
    print(f"{'='*90}")
    print(f"  {'Set':<8s}  {'DeMorgan':>12s}  {'Nusslein':>12s}  {'Chancellor':>12s}  {'ORchain':>12s}")
    print(f"  {'─'*65}")
    for r in all_results:
        def fmt(e): return f"{r[f'{e}_sa_solved']}/{r[f'{e}_pbit_solved']}"
        print(f"  {r['prefix']:<8s}  {fmt('DeMorgan'):>12s}  {fmt('Nusslein'):>12s}  {fmt('Chancellor'):>12s}  {fmt('ORchain'):>12s}")
