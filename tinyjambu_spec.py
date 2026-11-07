"""Independent word-level TinyJAMBU-128 (v2, NIST LWC finalist) written from the
specification, full AEAD encryption incl. associated data and partial blocks.
Used only to validate the experiment implementation against the official KAT."""
M32 = 0xFFFFFFFF

def _perm(s, k, rounds):
    s0, s1, s2, s3 = s
    for i in range(rounds // 32):
        t1 = ((s1 >> 15) | (s2 << 17)) & M32
        t2 = ((s2 >> 6) | (s3 << 26)) & M32
        t3 = ((s2 >> 21) | (s3 << 11)) & M32
        t4 = ((s2 >> 27) | (s3 << 5)) & M32
        fb = s0 ^ t1 ^ (~(t2 & t3) & M32) ^ t4 ^ k[i & 3]
        s0, s1, s2, s3 = s1, s2, s3, fb & M32
    return [s0, s1, s2, s3]

def _w(b):
    return int.from_bytes(b, "little")

def encrypt(key, nonce, ad, pt):
    k = [_w(key[i:i + 4]) for i in range(0, 16, 4)]
    s = _perm([0, 0, 0, 0], k, 1024)                      # key setup
    for i in range(3):                                    # nonce setup
        s[1] ^= 0x10; s = _perm(s, k, 640); s[3] ^= _w(nonce[4 * i:4 * i + 4])
    for i in range(0, len(ad) // 4 * 4, 4):               # associated data
        s[1] ^= 0x30; s = _perm(s, k, 640); s[3] ^= _w(ad[i:i + 4])
    r = len(ad) % 4
    if r:
        s[1] ^= 0x30; s = _perm(s, k, 640); s[3] ^= _w(ad[-r:]); s[1] ^= r
    ct = b""
    for i in range(0, len(pt) // 4 * 4, 4):               # message
        s[1] ^= 0x50; s = _perm(s, k, 1024); m = _w(pt[i:i + 4])
        s[3] ^= m; ct += ((s[2] ^ m) & M32).to_bytes(4, "little")
    r = len(pt) % 4
    if r:
        s[1] ^= 0x50; s = _perm(s, k, 1024); m = _w(pt[-r:])
        s[3] ^= m; ct += ((s[2] ^ m) & M32).to_bytes(4, "little")[:r]; s[1] ^= r
    s[1] ^= 0x70; s = _perm(s, k, 1024); t0 = s[2]       # finalization
    s[1] ^= 0x70; s = _perm(s, k, 640); t1 = s[2]
    return ct + t0.to_bytes(4, "little") + t1.to_bytes(4, "little")

def read_kat(path):
    vec, cur = [], {}
    for line in open(path):
        line = line.strip()
        if not line:
            if cur: vec.append(cur); cur = {}
            continue
        k, _, v = line.partition("=")
        cur[k.strip()] = v.strip()
    if cur: vec.append(cur)
    return vec
