import sys, time, random, json
sys.argv = [sys.argv[0]]
import numpy as np
import TinyJAMBU128v2_collision as tj
from tinyjambu_spec import encrypt
from tj_fast import setup_state, tags_range, words
random.seed(1); ok = 0; n = 0
for trial in range(1, 6):
    key, nonce = tj.make_key_nonce(20260726, trial)
    kw, nw = words(key), words(nonce); st = setup_state(kw, nw)
    a = tags_range(0, 2000, st, kw)
    b = tj.generate_tags_range(0, 2000, *[np.uint64(v) for v in kw], *[np.uint64(v) for v in nw])
    ok += int((a == b).sum()); n += 2000
    for c in random.sample(range(2000), 20):
        assert int(a[c]).to_bytes(8, "little") == encrypt(key, nonce, b"", c.to_bytes(8, "little"))[-8:]
print("fast == authors' implementation:", ok, "/", n)
key, nonce = tj.make_key_nonce(20260726, 1); kw, nw = words(key), words(nonce); st = setup_state(kw, nw)
tags_range(0, 10, st, kw)
t = time.perf_counter(); tags_range(0, 2_000_000, st, kw); dt = time.perf_counter() - t
print(f"fast single-core: {2e6/dt:,.0f} tags/s")
t = time.perf_counter(); tj.generate_tags_range(0, 1_000_000, *[np.uint64(v) for v in kw], *[np.uint64(v) for v in nw]); dt = time.perf_counter() - t
print(f"authors' numba generation only (incl. compile?): {1e6/dt:,.0f} tags/s")
