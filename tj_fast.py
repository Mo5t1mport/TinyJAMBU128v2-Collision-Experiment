"""Fast TinyJAMBU-128 tag computation for fixed (K, N), empty AD, 8-byte message.
The state after key and nonce setup does not depend on the message, so it is
computed once per (K, N); per message only 2 x P1024 (message) + P1024 + P640
(finalization) remain. Equality with the authors' implementation and with the
specification-based implementation is checked in validate_fast()."""
import numpy as np
from numba import njit, prange

M32 = np.uint64(0xFFFFFFFF)


@njit(cache=True, inline="always")
def _perm(s0, s1, s2, s3, k0, k1, k2, k3, rounds):
    for i in range(rounds // 32):
        r = i & 3
        kw = k0 if r == 0 else (k1 if r == 1 else (k2 if r == 2 else k3))
        t1 = (s1 >> np.uint64(15)) | (s2 << np.uint64(17))
        t2 = (s2 >> np.uint64(6)) | (s3 << np.uint64(26))
        t3 = (s2 >> np.uint64(21)) | (s3 << np.uint64(11))
        t4 = (s2 >> np.uint64(27)) | (s3 << np.uint64(5))
        fb = (s0 ^ t1 ^ (~(t2 & t3)) ^ t4 ^ kw) & M32
        s0, s1, s2, s3 = s1, s2, s3, fb
    return s0, s1, s2, s3


@njit(cache=True)
def setup_state(k, n):
    s0 = s1 = s2 = s3 = np.uint64(0)
    s0, s1, s2, s3 = _perm(s0, s1, s2, s3, k[0], k[1], k[2], k[3], 1024)
    for j in range(3):
        s1 ^= np.uint64(0x10)
        s0, s1, s2, s3 = _perm(s0, s1, s2, s3, k[0], k[1], k[2], k[3], 640)
        s3 ^= n[j]
    return np.array([s0, s1, s2, s3], dtype=np.uint64)


@njit(cache=True, inline="always")
def tag64(x, st, k):
    """Tag of the 8-byte little-endian message x (uint64) under the precomputed state."""
    s0, s1, s2, s3 = st[0], st[1], st[2], st[3]
    s1 ^= np.uint64(0x50)
    s0, s1, s2, s3 = _perm(s0, s1, s2, s3, k[0], k[1], k[2], k[3], 1024)
    s3 ^= x & M32
    s1 ^= np.uint64(0x50)
    s0, s1, s2, s3 = _perm(s0, s1, s2, s3, k[0], k[1], k[2], k[3], 1024)
    s3 ^= (x >> np.uint64(32)) & M32
    s1 ^= np.uint64(0x70)
    s0, s1, s2, s3 = _perm(s0, s1, s2, s3, k[0], k[1], k[2], k[3], 1024)
    lo = s2 & M32
    s1 ^= np.uint64(0x70)
    s0, s1, s2, s3 = _perm(s0, s1, s2, s3, k[0], k[1], k[2], k[3], 640)
    return lo | ((s2 & M32) << np.uint64(32))


@njit(cache=True)
def tags_range(start, count, st, k):
    out = np.empty(count, dtype=np.uint64)
    for i in range(count):
        out[i] = tag64(np.uint64(start + i), st, k)
    return out


def words(b):
    return np.array([int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)], dtype=np.uint64)
