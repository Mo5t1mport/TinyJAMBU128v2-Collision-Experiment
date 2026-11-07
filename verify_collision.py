"""Stand-alone check of the reported 64-bit tag collision with the
specification-based implementation (validated on all 1,089 KAT vectors)."""
from tinyjambu_spec import encrypt
K = bytes.fromhex("f218514a8a7a4f0167bf71e762aa82e6")
N = bytes.fromhex("1af5bd60a5a3c59439a18a82")
M1 = bytes.fromhex("8e9819a83f03e20a")
M2 = bytes.fromhex("82deab10e2781a0d")
c1, c2 = encrypt(K, N, b"", M1), encrypt(K, N, b"", M2)
print("C1||T1 =", c1.hex()); print("C2||T2 =", c2.hex())
print("M1 != M2:", M1 != M2, "| T1 == T2:", c1[-8:] == c2[-8:])
