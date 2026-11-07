import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": "#888",
                     "axes.linewidth": 0.6})
R = json.load(open("results/truncated_tags.json"))
COL = {"32": "#2a78d6", "36": "#eb6834", "40": "#1baf7a", "44": "#eda100"}
MK = {"32": "o", "36": "s", "40": "^", "44": "D"}
fig, ax = plt.subplots(figsize=(6.3, 3.6))
x = np.linspace(0, 3.2, 400)
ax.plot(x, 1 - np.exp(-x ** 2 / 2), color="#333", lw=1.6, label="Model birthday bound", zorder=3)
for t in ("32", "36", "40", "44"):
    q = np.sort(np.array(R[t]["first_collision_queries"], dtype=float)) / 2 ** (int(t) / 2)
    y = np.arange(1, len(q) + 1) / len(q)
    ax.step(q, y, where="post", color=COL[t], lw=1.0, alpha=0.9,
            label=f"t = {t} bit (100 trial)")
r48 = R["48"]
lo, hi = r48["rate_ci95"]
ax.errorbar([1.0], [r48["observed_rate_by_cap"]], yerr=[[r48["observed_rate_by_cap"] - lo], [hi - r48["observed_rate_by_cap"]]],
            fmt="o", color="#e87ba4", mec="white", ms=7, capsize=3, zorder=4,
            label="t = 48 bit, q = 2$^{24}$ (100 trial, IK 95%)")
ax.axvline(1.0, color="#bbb", lw=0.6, ls="--")
ax.set_xlabel("Jumlah kueri dinormalisasi, q / 2$^{t/2}$")
ax.set_ylabel("Proporsi trial dengan kolisi")
ax.set_xlim(0, 3.2); ax.set_ylim(0, 1.02)
ax.grid(color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig("results/fig2_truncated.png", dpi=300)
print("ok")
