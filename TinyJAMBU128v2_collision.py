from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# 1. Dependency
# ============================================================

REQUIRED = {
    "numpy": "numpy>=1.24",
    "pandas": "pandas>=2.0",
    "matplotlib": "matplotlib>=3.7",
    "numba": "numba>=0.59",
    "psutil": "psutil>=5.9",
}


def ensure_dependencies():
    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("Dependency belum tersedia:")
        for p in missing:
            print("  -", p)

        print("\nMeng-install dependency dengan pip...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing]
        )


ensure_dependencies()

import numpy as np
import pandas as pd
import psutil
from numba import njit


# ============================================================
# 2. Konstanta TinyJAMBU-128
# ============================================================

MASK32 = np.uint64(0xFFFFFFFF)

FRAME_NONCE = np.uint64(0x10)
FRAME_MSG = np.uint64(0x50)
FRAME_FINAL = np.uint64(0x70)

KEY_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 8

DEFAULT_MIN_EXP = 16
DEFAULT_MAX_EXP = 24
DEFAULT_TRIALS = 5

# Batas aman sesuai desain perangkat pengguna.
# 2^24 = 16,777,216.
HARD_MAX_EXP = 24


# ============================================================
# 3. TinyJAMBU-128
# ============================================================

@njit(cache=True, inline="always")
def _permute_128(s0, s1, s2, s3, k0, k1, k2, k3, rounds):
    loops = rounds // 32

    for i in range(loops):
        r = i & 3

        if r == 0:
            kw = k0
        elif r == 1:
            kw = k1
        elif r == 2:
            kw = k2
        else:
            kw = k3

        t1 = (s1 >> np.uint64(15)) | (s2 << np.uint64(17))
        t2 = (s2 >> np.uint64(6)) | (s3 << np.uint64(26))
        t3 = (s2 >> np.uint64(21)) | (s3 << np.uint64(11))
        t4 = (s2 >> np.uint64(27)) | (s3 << np.uint64(5))

        feedback = (
            s0
            ^ t1
            ^ (~(t2 & t3))
            ^ t4
            ^ kw
        ) & MASK32

        s0, s1, s2, s3 = s1, s2, s3, feedback

    return s0, s1, s2, s3


@njit(cache=True, inline="always")
def _tag_for_counter(counter, k0, k1, k2, k3, n0, n1, n2):
    """
    TinyJAMBU-128 tag untuk message 8-byte.

    Message:
        counter -> 8-byte little-endian

    AD:
        kosong

    Nonce:
        96-bit, tiga word 32-bit
    """

    s0 = np.uint64(0)
    s1 = np.uint64(0)
    s2 = np.uint64(0)
    s3 = np.uint64(0)

    # Key setup
    s0, s1, s2, s3 = _permute_128(
        s0, s1, s2, s3,
        k0, k1, k2, k3,
        1024
    )

    # Nonce setup
    s1 ^= FRAME_NONCE
    s0, s1, s2, s3 = _permute_128(
        s0, s1, s2, s3,
        k0, k1, k2, k3,
        640
    )
    s3 ^= n0

    s1 ^= FRAME_NONCE
    s0, s1, s2, s3 = _permute_128(
        s0, s1, s2, s3,
        k0, k1, k2, k3,
        640
    )
    s3 ^= n1

    s1 ^= FRAME_NONCE
    s0, s1, s2, s3 = _permute_128(
        s0, s1, s2, s3,
        k0, k1, k2, k3,
        640
    )
    s3 ^= n2

    # Message 8 byte = dua word 32-bit.
    lo = np.uint64(counter & np.uint64(0xFFFFFFFF))
    hi = np.uint64((counter >> np.uint64(32)) & np.uint64(0xFFFFFFFF))

    s1 ^= FRAME_MSG
    s0, s1, s2, s3 = _permute_128(
        s0, s1, s2, s3,
        k0, k1, k2, k3,
        1024
    )
    s3 ^= lo

    s1 ^= FRAME_MSG
    s0, s1, s2, s3 = _permute_128(
        s0, s1, s2, s3,
        k0, k1, k2, k3,
        1024
    )
    s3 ^= hi

    # Finalization
    s1 ^= FRAME_FINAL
    s0, s1, s2, s3 = _permute_128(
        s0, s1, s2, s3,
        k0, k1, k2, k3,
        1024
    )
    tag_lo = s2

    s1 ^= FRAME_FINAL
    s0, s1, s2, s3 = _permute_128(
        s0, s1, s2, s3,
        k0, k1, k2, k3,
        640
    )
    tag_hi = s2

    return tag_lo | (tag_hi << np.uint64(32))


@njit(cache=True)
def generate_tags_range(start_counter, num_queries,
                        k0, k1, k2, k3,
                        n0, n1, n2):
    """
    Menghasilkan tag untuk range counter.

    Digunakan secara batch agar tidak perlu menyimpan semua tag
    untuk eksperimen besar sekaligus.
    """
    tags = np.empty(num_queries, dtype=np.uint64)

    for i in range(num_queries):
        counter = np.uint64(start_counter + i)
        tags[i] = _tag_for_counter(
            counter,
            k0, k1, k2, k3,
            n0, n1, n2
        )

    return tags


# ============================================================
# 4. Reference implementation untuk validasi
# ============================================================

def _state_update_reference(state, key_int, round_index):
    s0 = (state >> 0) & 1
    s47 = (state >> 47) & 1
    s70 = (state >> 70) & 1
    s85 = (state >> 85) & 1
    s91 = (state >> 91) & 1

    ki = (key_int >> (round_index % 128)) & 1

    feedback = s0 ^ s47 ^ (1 ^ (s70 & s85)) ^ s91 ^ ki
    feedback &= 1

    state = (state >> 1) | (feedback << 127)

    return state & ((1 << 128) - 1)


def _permute_reference(state, key_int, rounds):
    for i in range(rounds):
        state = _state_update_reference(
            state, key_int, i
        )
    return state


def _tag_reference(message, key, nonce):
    key_int = int.from_bytes(key, "little")
    state = 0

    # Key setup
    state = _permute_reference(
        state, key_int, 1024
    )

    # Nonce
    for offset in (0, 4, 8):
        state ^= (0x10 << 32)
        state = _permute_reference(
            state, key_int, 640
        )

        word = int.from_bytes(
            nonce[offset:offset + 4],
            "little"
        )

        state ^= word << 96

    # Message: full 4-byte blocks.
    pos = 0

    while len(message) - pos >= 4:
        state ^= (0x50 << 32)
        state = _permute_reference(
            state, key_int, 1024
        )

        word = int.from_bytes(
            message[pos:pos + 4],
            "little"
        )

        state ^= word << 96
        pos += 4

    # Partial block.
    rem = len(message) - pos

    if rem:
        state ^= (0x50 << 32)
        state = _permute_reference(
            state, key_int, 1024
        )

        part = message[pos:]
        word = int.from_bytes(
            part + b"\x00" * (4 - rem),
            "little"
        )

        state ^= word << 96
        state ^= rem << 32

    # Finalization
    state ^= (0x70 << 32)
    state = _permute_reference(
        state, key_int, 1024
    )

    tag_lo = (state >> 64) & 0xFFFFFFFF

    state ^= (0x70 << 32)
    state = _permute_reference(
        state, key_int, 640
    )

    tag_hi = (state >> 64) & 0xFFFFFFFF

    tag = tag_lo | (tag_hi << 32)

    return tag.to_bytes(8, "little")


def _optimized_single_tag(message, key, nonce):
    if len(message) != 8:
        raise ValueError(
            "Optimized validation menggunakan message 8-byte."
        )

    kw = [
        int.from_bytes(key[i:i + 4], "little")
        for i in range(0, 16, 4)
    ]

    nw = [
        int.from_bytes(nonce[i:i + 4], "little")
        for i in range(0, 12, 4)
    ]

    counter = int.from_bytes(
        message, "little"
    )

    tag = _tag_for_counter(
        np.uint64(counter),
        np.uint64(kw[0]),
        np.uint64(kw[1]),
        np.uint64(kw[2]),
        np.uint64(kw[3]),
        np.uint64(nw[0]),
        np.uint64(nw[1]),
        np.uint64(nw[2]),
    )

    return int(tag).to_bytes(8, "little")


def validate_implementation(key, nonce):
    print("\n" + "=" * 70)
    print("VALIDASI IMPLEMENTASI")
    print("=" * 70)

    # Trigger Numba compilation.
    _ = generate_tags_range(
        0, 1,
        np.uint64(int.from_bytes(key[0:4], "little")),
        np.uint64(int.from_bytes(key[4:8], "little")),
        np.uint64(int.from_bytes(key[8:12], "little")),
        np.uint64(int.from_bytes(key[12:16], "little")),
        np.uint64(int.from_bytes(nonce[0:4], "little")),
        np.uint64(int.from_bytes(nonce[4:8], "little")),
        np.uint64(int.from_bytes(nonce[8:12], "little")),
    )

    messages = [
        b"\x00" * 8,
        bytes.fromhex("0001020304050607"),
        bytes.fromhex("08090a0b0c0d0e0f"),
        bytes.fromhex("ffffffffffffffff"),
        bytes.fromhex("1020304050607080"),
    ]

    for msg in messages:
        ref = _tag_reference(
            msg, key, nonce
        )

        opt = _optimized_single_tag(
            msg, key, nonce
        )

        if ref != opt:
            raise RuntimeError(
                "\nVALIDASI GAGAL!\n"
                f"Message   : {msg.hex()}\n"
                f"Reference : {ref.hex()}\n"
                f"Optimized : {opt.hex()}\n"
            )

    print("✓ Reference dan optimized implementation cocok.")
    print("✓ 5 message deterministik berhasil divalidasi.")
    print(
        "Catatan: untuk paper final, cocokkan juga dengan KAT "
        "resmi TinyJAMBU."
    )


# ============================================================
# 5. Probabilitas birthday
# ============================================================

def birthday_probability(q, tag_bits=64):
    if q <= 1:
        return 0.0

    exponent = (
        -q * (q - 1)
        / (2.0 * (2.0 ** tag_bits))
    )

    return -math.expm1(exponent)


def birthday_q50(tag_bits=64):
    return math.sqrt(
        2.0
        * (2.0 ** tag_bits)
        * math.log(2.0)
    )


# ============================================================
# 6. Key + nonce reproducible
# ============================================================

def make_key_nonce(seed, trial):
    base = (
        f"TINYJAMBU-V2-{seed}-{trial}"
    ).encode()

    digest = hashlib.sha256(
        base
    ).digest()

    key = digest[:16]

    nonce_digest = hashlib.sha256(
        digest + b"-NONCE"
    ).digest()

    nonce = nonce_digest[:12]

    return key, nonce


# ============================================================
# 7. Collision search berbasis batch
# ============================================================

def search_collision_batched(
    q,
    key,
    nonce,
    batch_size,
    tag_bits=64,
):
    """
    Pencarian collision secara batch.

    Untuk q <= 2^24, default menggunakan sorting per batch.
    Batch membantu mengendalikan peak memory, tetapi collision yang
    melibatkan dua batch berbeda hanya dapat dideteksi jika kita
    mempertahankan struktur indeks tag antar-batch.

    Karena tujuan utama eksperimen adalah q <= 2^24 dan perangkat
    terbatas, implementasi ini menggunakan satu tabel tag Python
    untuk collision search lintas-batch. Ini lebih hemat peak memory
    daripada menampung semua tag NumPy sekaligus, tetapi tetap
    membutuhkan RAM yang cukup.

    Jika memory tidak mencukupi, program dapat dihentikan dengan
    --skip-collision-search untuk hanya mengukur performa.
    """

    kw = [
        np.uint64(int.from_bytes(key[0:4], "little")),
        np.uint64(int.from_bytes(key[4:8], "little")),
        np.uint64(int.from_bytes(key[8:12], "little")),
        np.uint64(int.from_bytes(key[12:16], "little")),
    ]

    nw = [
        np.uint64(int.from_bytes(nonce[0:4], "little")),
        np.uint64(int.from_bytes(nonce[4:8], "little")),
        np.uint64(int.from_bytes(nonce[8:12], "little")),
    ]

    # Untuk tag 64-bit, dictionary key menggunakan integer Python.
    # Ini tetap memory-intensive pada q besar, tetapi peak NumPy
    # batch terkontrol.
    seen = {}

    mask = (
        (1 << tag_bits) - 1
        if tag_bits < 64
        else None
    )

    for start in range(0, q, batch_size):
        count = min(
            batch_size,
            q - start
        )

        tags = generate_tags_range(
            start,
            count,
            kw[0], kw[1],
            kw[2], kw[3],
            nw[0], nw[1], nw[2],
        )

        for local_i, tag_np in enumerate(tags):
            query = start + local_i
            tag = int(tag_np)

            if mask is not None:
                tag_cmp = tag & mask
            else:
                tag_cmp = tag

            previous = seen.get(
                tag_cmp
            )

            if previous is not None:
                # Message = query encoded 8-byte LE.
                return {
                    "collision_found": True,
                    "collision_query_1": previous,
                    "collision_query_2": query,
                    "message_1": previous.to_bytes(
                        8, "little"
                    ).hex(),
                    "message_2": query.to_bytes(
                        8, "little"
                    ).hex(),
                    "collision_tag": tag.to_bytes(
                        8, "little"
                    ).hex(),
                }

            seen[tag_cmp] = query

        del tags

    return {
        "collision_found": False,
        "collision_query_1": "",
        "collision_query_2": "",
        "message_1": "",
        "message_2": "",
        "collision_tag": "",
    }


# ============================================================
# 8. Satu trial
# ============================================================

def run_trial(
    q,
    key,
    nonce,
    batch_size,
    search_collision=True,
    tag_bits=64,
):
    kw = [
        np.uint64(int.from_bytes(key[0:4], "little")),
        np.uint64(int.from_bytes(key[4:8], "little")),
        np.uint64(int.from_bytes(key[8:12], "little")),
        np.uint64(int.from_bytes(key[12:16], "little")),
    ]

    nw = [
        np.uint64(int.from_bytes(nonce[0:4], "little")),
        np.uint64(int.from_bytes(nonce[4:8], "little")),
        np.uint64(int.from_bytes(nonce[8:12], "little")),
    ]

    process = psutil.Process(
        os.getpid()
    )

    rss_before = (
        process.memory_info().rss
    )

    start = time.perf_counter()

    # Jika collision search aktif, gunakan batch dictionary.
    # Jika tidak, hanya benchmark generation.
    if search_collision:
        collision = search_collision_batched(
            q=q,
            key=key,
            nonce=nonce,
            batch_size=batch_size,
            tag_bits=tag_bits,
        )
    else:
        # Benchmark generation saja.
        for start_counter in range(
            0, q, batch_size
        ):
            count = min(
                batch_size,
                q - start_counter
            )

            tags = generate_tags_range(
                start_counter,
                count,
                kw[0], kw[1],
                kw[2], kw[3],
                nw[0], nw[1], nw[2],
            )

            del tags

        collision = {
            "collision_found": False,
            "collision_query_1": "",
            "collision_query_2": "",
            "message_1": "",
            "message_2": "",
            "collision_tag": "",
        }

    elapsed = (
        time.perf_counter() - start
    )

    rss_after = (
        process.memory_info().rss
    )

    return {
        "queries": q,
        "elapsed_seconds": elapsed,
        "throughput_queries_per_sec": (
            q / elapsed
            if elapsed > 0
            else float("inf")
        ),
        "rss_before_mb": rss_before / 1024**2,
        "rss_after_mb": rss_after / 1024**2,
        "rss_delta_mb": (
            rss_after - rss_before
        ) / 1024**2,
        **collision,
        "theoretical_collision_probability": (
            birthday_probability(
                q, 64
            )
        ),
    }


# ============================================================
# 9. Grafik
# ============================================================

def make_plots(
    trial_df,
    summary_df,
    output_dir,
):
    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))

    for trial in sorted(
        trial_df["trial"].unique()
    ):
        d = trial_df[
            trial_df["trial"] == trial
        ]

        plt.plot(
            d["queries"],
            d["elapsed_seconds"],
            marker="o",
            alpha=0.55,
            label=f"Trial {trial}",
        )

    plt.xscale(
        "log",
        base=2
    )

    plt.xlabel(
        "Number of queries"
    )

    plt.ylabel(
        "Execution time (seconds)"
    )

    plt.title(
        "TinyJAMBU-128: Query Count vs Execution Time"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.25
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "01_runtime_vs_queries.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        summary_df["queries"],
        summary_df["mean_throughput"],
        yerr=summary_df[
            "std_throughput"
        ].fillna(0),
        marker="o",
        capsize=4,
    )

    plt.xscale(
        "log",
        base=2
    )

    plt.xlabel(
        "Number of queries"
    )

    plt.ylabel(
        "Throughput (queries/second)"
    )

    plt.title(
        "TinyJAMBU-128: Mean Throughput ± SD"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "02_throughput_vs_queries.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        summary_df["queries"],
        summary_df[
            "theoretical_collision_probability"
        ],
        marker="o",
    )

    plt.xscale(
        "log",
        base=2
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Number of queries"
    )

    plt.ylabel(
        "Theoretical collision probability"
    )

    plt.title(
        "64-bit Tag: Birthday Collision Probability"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "03_theoretical_probability.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Observed collision
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))

    observed = (
        summary_df[
            "collision_trials"
        ]
        / summary_df[
            "number_of_trials"
        ]
    )

    plt.plot(
        summary_df["queries"],
        observed,
        marker="o",
    )

    plt.xscale(
        "log",
        base=2
    )

    plt.ylim(
        -0.05,
        1.05
    )

    plt.xlabel(
        "Number of queries"
    )

    plt.ylabel(
        "Observed collision rate"
    )

    plt.title(
        "Observed Collision Rate"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "04_observed_collision.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# 10. Summary
# ============================================================

def summarize(
    trial_df,
):
    rows = []

    for q in sorted(
        trial_df["queries"].unique()
    ):
        d = trial_df[
            trial_df["queries"] == q
        ]

        rows.append({
            "queries": q,
            "log2_queries": int(
                math.log2(q)
            ),
            "number_of_trials": len(d),

            "mean_runtime_seconds": (
                d["elapsed_seconds"]
                .mean()
            ),

            "median_runtime_seconds": (
                d["elapsed_seconds"]
                .median()
            ),

            "std_runtime_seconds": (
                d["elapsed_seconds"]
                .std(ddof=1)
                if len(d) > 1
                else 0.0
            ),

            "mean_throughput": (
                d[
                    "throughput_queries_per_sec"
                ].mean()
            ),

            "median_throughput": (
                d[
                    "throughput_queries_per_sec"
                ].median()
            ),

            "std_throughput": (
                d[
                    "throughput_queries_per_sec"
                ].std(ddof=1)
                if len(d) > 1
                else 0.0
            ),

            "mean_rss_delta_mb": (
                d["rss_delta_mb"].mean()
            ),

            "max_rss_after_mb": (
                d["rss_after_mb"].max()
            ),

            "collision_trials": int(
                d[
                    "collision_found"
                ].sum()
            ),

            "collision_rate_observed": (
                d[
                    "collision_found"
                ].mean()
            ),

            "theoretical_collision_probability": (
                birthday_probability(
                    q, 64
                )
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 11. Konfigurasi
# ============================================================

def write_config(
    path,
    args,
    system,
):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "TinyJAMBU-128 Collision Experiment v2\n"
        )
        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Minimum exponent : {args.min_exp}\n"
        )

        f.write(
            f"Maximum exponent : {args.max_exp}\n"
        )

        f.write(
            f"Trials            : {args.trials}\n"
        )

        f.write(
            f"Batch size        : {args.batch_size:,}\n"
        )

        f.write(
            f"Seed              : {args.seed}\n"
        )

        f.write(
            f"Collision search   : "
            f"{not args.skip_collision_search}\n"
        )

        f.write(
            "Tag size           : 64 bit\n"
        )

        f.write(
            "Key size           : 128 bit\n"
        )

        f.write(
            "Nonce size         : 96 bit\n"
        )

        f.write(
            "Associated Data    : empty\n"
        )

        f.write(
            "Message size       : 8 bytes\n\n"
        )

        f.write(
            "Empirical maximum  : 2^24\n"
        )

        f.write(
            "Theoretical bound  : order 2^32\n"
        )

        f.write(
            "q50 for 64-bit tag : "
            f"{birthday_q50(64):,.0f}\n\n"
        )

        f.write(
            "System information\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        for k, v in system.items():
            f.write(
                f"{k}: {v}\n"
            )


def get_system_info():
    vm = psutil.virtual_memory()

    return {
        "Python": platform.python_version(),
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "CPU count": os.cpu_count(),
        "RAM total GB": round(
            vm.total / 1024**3,
            2,
        ),
        "RAM available GB": round(
            vm.available / 1024**3,
            2,
        ),
    }


# ============================================================
# 12. Argument
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "TinyJAMBU-128 collision experiment "
            "v2 for constrained hardware."
        )
    )

    parser.add_argument(
        "--min-exp",
        type=int,
        default=DEFAULT_MIN_EXP,
        help=(
            "Minimum q exponent. "
            "Default = 16."
        ),
    )

    parser.add_argument(
        "--max-exp",
        type=int,
        default=DEFAULT_MAX_EXP,
        help=(
            "Maximum q exponent. "
            "Default = 24. "
            "Hard maximum = 24."
        ),
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=(
            "Number of independent trials. "
            "Default = 5."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=262144,
        help=(
            "Batch size for tag generation. "
            "Default = 262144 (2^18)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260726,
        help=(
            "Reproducibility seed."
        ),
    )

    parser.add_argument(
        "--output",
        default="tinyjambu_v2_results",
        help=(
            "Output directory."
        ),
    )

    parser.add_argument(
        "--skip-collision-search",
        action="store_true",
        help=(
            "Only benchmark tag generation; "
            "does not retain collision dictionary."
        ),
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Skip reference implementation validation."
        ),
    )

    return parser.parse_args()


# ============================================================
# 13. Main
# ============================================================

def main():
    args = parse_args()

    # Hard safety limit.
    if args.max_exp > HARD_MAX_EXP:
        raise SystemExit(
            "\nDIBATALKAN: max-exp tidak boleh > 24 "
            "untuk versi perangkat terbatas ini.\n"
            "Gunakan --max-exp 24 atau lebih kecil."
        )

    if args.min_exp > args.max_exp:
        raise SystemExit(
            "min-exp harus <= max-exp."
        )

    if args.trials < 1:
        raise SystemExit(
            "trials minimal 1."
        )

    if args.batch_size < 1:
        raise SystemExit(
            "batch-size minimal 1."
        )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    system = get_system_info()

    print("\n" + "=" * 70)
    print(
        "TinyJAMBU-128 Collision Experiment v2"
    )
    print("=" * 70)

    print(
        "\nKonfigurasi:"
    )

    print(
        f"  q minimum      : 2^{args.min_exp}"
    )

    print(
        f"  q maximum      : 2^{args.max_exp}"
    )

    print(
        f"  trials         : {args.trials}"
    )

    print(
        f"  batch size     : {args.batch_size:,}"
    )

    print(
        "  tag            : 64-bit"
    )

    print(
        "  key            : 128-bit"
    )

    print(
        "  nonce          : 96-bit (fixed/trial)"
    )

    print(
        "  AD             : empty"
    )

    print(
        "  message        : 8-byte counter"
    )

    print(
        f"  output         : {output_dir.resolve()}"
    )

    print(
        "\nBatas empiris dikunci pada 2^24 "
        "agar tidak melewati kemampuan perangkat."
    )

    print(
        "\nSystem:"
    )

    for k, v in system.items():
        print(
            f"  {k}: {v}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    key0, nonce0 = make_key_nonce(
        args.seed,
        0,
    )

    if not args.skip_validation:
        validate_implementation(
            key0,
            nonce0,
        )

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(
        "EKSPERIMEN"
    )
    print("=" * 70)

    all_rows = []

    for trial in range(
        1,
        args.trials + 1,
    ):
        key, nonce = make_key_nonce(
            args.seed,
            trial,
        )

        print(
            f"\nTrial {trial}/{args.trials}"
        )

        print(
            f"  key   = {key.hex()}"
        )

        print(
            f"  nonce = {nonce.hex()}"
        )

        for exp in range(
            args.min_exp,
            args.max_exp + 1,
        ):
            q = 2 ** exp

            print(
                f"\n  q = 2^{exp} "
                f"({q:,})"
            )

            print(
                "  Running...",
                end=" ",
                flush=True,
            )

            result = run_trial(
                q=q,
                key=key,
                nonce=nonce,
                batch_size=args.batch_size,
                search_collision=(
                    not args.skip_collision_search
                ),
                tag_bits=64,
            )

            print(
                f"{result['elapsed_seconds']:.3f} s"
            )

            print(
                "  Throughput:",
                f"{result['throughput_queries_per_sec']:,.0f}",
                "query/s"
            )

            print(
                "  Collision:",
                "YES"
                if result["collision_found"]
                else "NO",
            )

            if result[
                "collision_found"
            ]:
                print(
                    "  Collision pair:",
                    result[
                        "collision_query_1"
                    ],
                    "and",
                    result[
                        "collision_query_2"
                    ],
                )

            row = {
                "trial": trial,
                "seed": args.seed,
                "log2_queries": exp,
                "queries": q,
                "tag_bits": 64,
                "key_hex": key.hex(),
                "nonce_hex": nonce.hex(),
                **result,
            }

            all_rows.append(row)

    # --------------------------------------------------------
    # Dataframe
    # --------------------------------------------------------
    trial_df = pd.DataFrame(
        all_rows
    )

    summary_df = summarize(
        trial_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    trial_csv = (
        output_dir
        / "results_per_trial.csv"
    )

    summary_csv = (
        output_dir
        / "summary_statistics.csv"
    )

    trial_df.to_csv(
        trial_csv,
        index=False,
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    write_config(
        output_dir
        / "experiment_config.txt",
        args,
        system,
    )

    make_plots(
        trial_df,
        summary_df,
        output_dir,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    total_points = len(
        trial_df
    )

    total_collisions = int(
        trial_df[
            "collision_found"
        ].sum()
    )

    max_q = int(
        trial_df[
            "queries"
        ].max()
    )

    p_max = birthday_probability(
        max_q,
        64,
    )

    print("\n" + "=" * 70)
    print(
        "EKSPERIMEN SELESAI"
    )
    print("=" * 70)

    print(
        f"\nTotal experiment-points : "
        f"{total_points}"
    )

    print(
        f"Observed 64-bit collisions : "
        f"{total_collisions}"
    )

    print(
        f"Maximum q : {max_q:,}"
    )

    print(
        f"Theoretical P(collision) "
        f"at maximum q : {p_max:.12g}"
    )

    print(
        f"64-bit q50 birthday point : "
        f"{birthday_q50(64):,.0f}"
    )

    print(
        "\nFile hasil:"
    )

    print(
        f"  {trial_csv}"
    )

    print(
        f"  {summary_csv}"
    )

    print(
        f"  {output_dir / 'experiment_config.txt'}"
    )

    print(
        f"  {output_dir / '01_runtime_vs_queries.png'}"
    )

    print(
        f"  {output_dir / '02_throughput_vs_queries.png'}"
    )

    print(
        f"  {output_dir / '03_theoretical_probability.png'}"
    )

    print(
        f"  {output_dir / '04_observed_collision.png'}"
    )

    print(
        "\nCatatan untuk paper:"
    )

    print(
        "  Hasil empiris berhenti pada 2^24."
    )

    print(
        "  2^32 hanya digunakan sebagai "
        "birthday-bound/theoretical reference."
    )

    print(
        "  Tidak ditemukan collision bukan "
        "berarti collision tidak mungkin."
    )


if __name__ == "__main__":
    main()
