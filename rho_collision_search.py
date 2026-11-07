"""Memory-efficient search for a full 64-bit tag collision of TinyJAMBU-128
(fixed key and nonce, empty AD, 8-byte messages) with the parallel
distinguished-point method of van Oorschot and Wiener.

f(x) = Tag_{K,N}(empty AD, LE64(x)) maps 64-bit values to 64-bit values, so a
collision of f is a pair of distinct 8-byte messages with identical tags under
the same key and nonce.  Chains x -> f(x) -> ... are iterated until a
distinguished point (low D bits zero) is reached; only (DP, start, length) is
stored.  Two chains ending in the same DP have merged, and walking them in step
reveals the colliding pair.  Expected work ~ sqrt(pi/2 * 2^64) ~ 5.4e9 tags.
"""
import json, sys, time, platform, os
import numpy as np
from numba import njit, prange, get_num_threads

sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
SEED = 20260726          # same seed as TinyJAMBU128v2_collision.py
TRIAL = 1                # key/nonce of trial 1
D = 20                   # distinguished-point bits
MAXLEN = 20 * (1 << D)   # abandon chains that are too long (cycles)
BATCH = 64               # chains per parallel batch
START_SEED = 20261107    # seed for chain starting points

import importlib.util
spec = importlib.util.spec_from_file_location("tjc", os.path.join(os.path.dirname(__file__), "TinyJAMBU128v2_collision.py"))
_argv = sys.argv; sys.argv = [sys.argv[0]]
tjc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tjc); sys.argv = _argv
from tj_fast import setup_state, tag64, words
from tinyjambu_spec import encrypt


@njit(parallel=True, cache=True)
def run_chains(starts, st, k, dmask, maxlen, out_dp, out_len):
    for i in prange(starts.shape[0]):
        x = starts[i]
        L = 0
        while True:
            x = tag64(x, st, k)
            L += 1
            if (x & dmask) == 0:
                break
            if L >= maxlen:
                L = -1
                break
        out_dp[i] = x
        out_len[i] = L


@njit(cache=True)
def walk(x, n, st, k):
    for _ in range(n):
        x = tag64(x, st, k)
    return x


@njit(cache=True)
def find_merge(a, b, limit, st, k):
    """a and b are aligned (same distance to the common DP). Returns (x, y, ok)."""
    if a == b:
        return a, b, False
    for _ in range(limit):
        fa = tag64(a, st, k)
        fb = tag64(b, st, k)
        if fa == fb:
            return a, b, True
        a, b = fa, fb
    return a, b, False


def main():
    key, nonce = tjc.make_key_nonce(SEED, TRIAL)
    kw, nw = words(key), words(nonce)
    st = setup_state(kw, nw)
    dmask = np.uint64((1 << D) - 1)
    rng = np.random.Generator(np.random.PCG64(START_SEED))
    table = {}
    evals = 0
    chains = 0
    t0 = time.perf_counter()
    out_dp = np.empty(BATCH, dtype=np.uint64)
    out_len = np.empty(BATCH, dtype=np.int64)
    while True:
        starts = rng.integers(0, 2**64, size=BATCH, dtype=np.uint64, endpoint=False)
        run_chains(starts, st, kw, dmask, MAXLEN, out_dp, out_len)
        for s, dp, L in zip(starts.tolist(), out_dp.tolist(), out_len.tolist()):
            chains += 1
            if L < 0:
                evals += MAXLEN
                continue
            evals += L
            if dp in table and table[dp][0] != s:
                s2, L2 = table[dp]
                a, la, b, lb = (s, L, s2, L2)
                if la < lb:
                    a, la, b, lb = b, lb, a, la
                a = walk(np.uint64(a), la - lb, st, kw)
                x, y, ok = find_merge(np.uint64(a), np.uint64(b), lb + 1, st, kw)
                evals += 2 * lb
                if ok:
                    x, y = int(x), int(y)
                    elapsed = time.perf_counter() - t0
                    m1, m2 = x.to_bytes(8, "little"), y.to_bytes(8, "little")
                    t1 = encrypt(key, nonce, b"", m1)[-8:]
                    t2 = encrypt(key, nonce, b"", m2)[-8:]
                    o1 = tjc._optimized_single_tag(m1, key, nonce)
                    o2 = tjc._optimized_single_tag(m2, key, nonce)
                    res = {
                        "key_hex": key.hex(), "nonce_hex": nonce.hex(), "ad": "",
                        "message_1_hex": m1.hex(), "message_2_hex": m2.hex(),
                        "tag_1_hex": t1.hex(), "tag_2_hex": t2.hex(),
                        "ciphertext_1_hex": encrypt(key, nonce, b"", m1)[:8].hex(),
                        "ciphertext_2_hex": encrypt(key, nonce, b"", m2)[:8].hex(),
                        "tags_equal_spec_impl": t1 == t2, "tags_equal_authors_impl": o1 == o2 == t1,
                        "messages_distinct": m1 != m2,
                        "tag_evaluations": evals, "log2_evaluations": float(np.log2(evals)),
                        "chains": chains, "distinguished_points_stored": len(table),
                        "dp_bits": D, "elapsed_seconds": elapsed, "threads": get_num_threads(),
                        "python": platform.python_version(), "platform": platform.platform(),
                        "cpu": "see report", "start_seed": START_SEED, "key_seed": SEED, "trial": TRIAL,
                    }
                    json.dump(res, open("results/rho_collision.json", "w"), indent=1)
                    print(json.dumps(res, indent=1), flush=True)
                    return
            table[dp] = (s, L)
        el = time.perf_counter() - t0
        print(f"chains={chains} evals={evals:.3e} (2^{np.log2(evals):.2f}) dps={len(table)} "
              f"rate={evals/el:,.0f}/s t={el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
