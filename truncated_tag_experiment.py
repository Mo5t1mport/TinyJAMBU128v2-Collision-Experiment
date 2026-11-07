"""Empirical validation of the generic birthday model on TinyJAMBU-128 tags
truncated to t bits (t = 32, 36, 40, 44, 48).

For each t and each trial (independent key/nonce from make_key_nonce(seed, trial)
of TinyJAMBU128v2_collision.py), tags of the 8-byte counter messages 0, 1, 2, ...
are generated under a fixed nonce and empty AD until the first pair of messages
with equal t-bit tags (the t least significant bits of the 64-bit tag) appears,
or until the cap of 2^24 queries, the same cap as the 64-bit experiment.
The query count Q at the first collision is compared with the birthday model
P(Q <= q) = 1 - exp(-q(q-1)/2^(t+1)).
"""
import json, math, os, sys, time
import numpy as np
from numba import njit, prange
from scipy import stats

import importlib.util
spec = importlib.util.spec_from_file_location("tjc", os.path.join(os.path.dirname(__file__), "TinyJAMBU128v2_collision.py"))
_argv = sys.argv; sys.argv = [sys.argv[0]]
tjc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tjc); sys.argv = _argv
from tj_fast import setup_state, tag64, words

SEED = 20260726
TRIALS = 100
T_BITS = [32, 36, 40, 44, 48]
CAP = 1 << 24
TBL_BITS = 25                      # open-addressing table of 2^25 slots (load <= 0.5)


@njit(cache=True)
def first_collision(st, k, t, cap, table):
    """Returns the 1-based index of the query that produced the first repeated
    t-bit tag, or 0 if none occurs within cap queries."""
    mask = np.uint64((1 << t) - 1) if t < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
    size = table.shape[0]
    shift = np.uint64(64 - TBL_BITS)
    table[:] = 0
    for i in range(cap):
        v = (tag64(np.uint64(i), st, k) & mask) + np.uint64(1)     # 0 marks an empty slot
        h = (v * np.uint64(0x9E3779B97F4A7C15)) >> shift
        while True:
            cur = table[h]
            if cur == 0:
                table[h] = v
                break
            if cur == v:
                return i + 1
            h = (h + np.uint64(1)) & np.uint64(size - 1)
    return 0


@njit(parallel=True, cache=True)
def run_trials(states, keys, t, cap, out):
    n = states.shape[0]
    nthreads = 2
    tables = np.zeros((nthreads, 1 << TBL_BITS), dtype=np.uint64)
    for w in prange(nthreads):
        for j in range(w, n, nthreads):
            out[j] = first_collision(states[j], keys[j], t, cap, tables[w])


def cdf(q, t):
    return -math.expm1(-q * (q - 1) / 2.0 ** (t + 1))


def main():
    states = np.empty((TRIALS, 4), dtype=np.uint64)
    keys = np.empty((TRIALS, 4), dtype=np.uint64)
    for j in range(TRIALS):
        key, nonce = tjc.make_key_nonce(SEED, j + 1)
        keys[j] = words(key)
        states[j] = setup_state(words(key), words(nonce))
    results = {}
    for t in T_BITS:
        out = np.zeros(TRIALS, dtype=np.int64)
        t0 = time.perf_counter()
        run_trials(states, keys, t, CAP, out)
        el = time.perf_counter() - t0
        found = out[out > 0]
        n_found = int(found.size)
        N = 2.0 ** t
        exp_mean = math.sqrt(math.pi * N / 2) + 2 / 3
        p_cap = cdf(CAP, t)
        lo, hi = stats.binomtest(n_found, TRIALS).proportion_ci(0.95, method="exact")
        r = {"t": t, "trials": TRIALS, "cap": CAP, "collisions": n_found,
             "observed_rate_by_cap": n_found / TRIALS, "rate_ci95": [lo, hi],
             "theory_rate_by_cap": p_cap, "elapsed_seconds": el,
             "first_collision_queries": out.tolist()}
        # rate of trials with a collision by q = 2^e, e = 16..24, vs theory
        r["by_exponent"] = [{"e": e, "observed": float(np.mean((out > 0) & (out <= 2 ** e))),
                             "theory": cdf(2 ** e, t)} for e in range(16, 25)]
        if n_found == TRIALS:          # uncensored: compare full distribution
            r["mean_Q"] = float(found.mean()); r["sd_Q"] = float(found.std(ddof=1))
            r["median_Q"] = float(np.median(found))
            r["theory_mean_Q"] = exp_mean
            r["theory_median_Q"] = math.sqrt(2 * math.log(2) * N)
            r["ks"] = stats.kstest(found, lambda q: np.array([cdf(x, t) for x in np.atleast_1d(q)])).pvalue
        else:
            # censored at CAP: compare the proportion with the binomial test
            r["binom_pvalue"] = stats.binomtest(n_found, TRIALS, p_cap).pvalue
        results[t] = r
        print({k: v for k, v in r.items() if k not in ("first_collision_queries", "by_exponent")}, flush=True)
    json.dump(results, open("results/truncated_tags.json", "w"), indent=1)


if __name__ == "__main__":
    main()
