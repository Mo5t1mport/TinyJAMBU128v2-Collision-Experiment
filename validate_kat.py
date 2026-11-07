"""Validates (1) the independent spec implementation on all official KAT vectors and
(2) the authors' Numba implementation on every KAT vector with an 8-byte message
and empty AD, plus 10,000 random key/nonce/message triples against (1)."""
import json, random, sys
sys.argv = [sys.argv[0]]
from tinyjambu_spec import encrypt, read_kat
import TinyJAMBU128v2_collision as tj

vec = read_kat("LWC_AEAD_KAT_128_96_TinyJAMBU-128.txt")
ok = sum(encrypt(bytes.fromhex(v["Key"]), bytes.fromhex(v["Nonce"]), bytes.fromhex(v["AD"]),
                 bytes.fromhex(v["PT"])).hex().upper() == v["CT"] for v in vec)
print(f"spec implementation: {ok}/{len(vec)} official KAT vectors")
sel = [v for v in vec if len(v["PT"]) == 16 and v["AD"] == ""]
ok2 = sum(tj._optimized_single_tag(bytes.fromhex(v["PT"]), bytes.fromhex(v["Key"]),
                                   bytes.fromhex(v["Nonce"])).hex().upper() == v["CT"][-16:] for v in sel)
print(f"authors' Numba tag: {ok2}/{len(sel)} KAT vectors with 8-byte PT and empty AD")
random.seed(20260726); ok3 = 0
for _ in range(10000):
    key, nonce, msg = (random.randbytes(n) for n in (16, 12, 8))
    ok3 += tj._optimized_single_tag(msg, key, nonce) == encrypt(key, nonce, b"", msg)[-8:]
print(f"authors' Numba tag vs spec implementation: {ok3}/10000 random inputs")
json.dump({"kat_total": len(vec), "kat_pass_spec": ok, "kat_8B_noAD": len(sel), "kat_pass_numba": ok2,
           "random_pass": ok3}, open("results/kat_validation.json", "w"), indent=1)
