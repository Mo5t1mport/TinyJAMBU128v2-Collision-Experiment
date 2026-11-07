import resource, random, json, sys
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
random.seed(1); n = 1 << 24
seen = {}
for i in range(n):
    seen[random.getrandbits(64)] = i
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
r = {"entries": n, "peak_rss_MiB": peak / 1024, "baseline_MiB": base / 1024,
     "dict_MiB": (peak - base) / 1024, "bytes_per_entry": (peak - base) * 1024 / n}
print(r); json.dump(r, open("results/dict_memory.json", "w"), indent=1)
