"""Accuracy vs. ViT-encoder compute-savings plot for the Discussion section's
argument: pixel-level scoring (can skip encoding most candidate frames) and
post-encoding scoring (SCSampler-style, must encode everything first) occupy
two categorically different regimes, not points on one smooth frontier.
All numbers are from Table tab:flops (compute savings) and Table tab:main
(macro-Jaccard mean +/- std, Cholec80 main setting).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 11})

# (label, compute_savings_pct, jaccard_mean, jaccard_std, is_headline, is_pixel_level)
points = [
    ("Uniform",              0.0, 0.7147, 0.0186, False, False),
    ("RandomSelect",        50.0, 0.7466, 0.0160, False, True),
    ("MobileNet-bias",      48.5, 0.7403, 0.0157, False, True),
    ("MGSampler",           50.0, 0.7509, 0.0182, False, True),
    ("Fixed-position",      50.0, 0.7508, 0.0163, True,  True),
    ("MobileNet-TS+",       48.5, 0.7543, 0.0170, True,  True),
    ("SCSampler (stoch.)",   0.0, 0.7516, 0.0193, False, False),
    ("SCSampler-argmax",     0.0, 0.7565, 0.0159, True,  False),
]

fig, ax = plt.subplots(figsize=(8.6, 5.2))
fig.subplots_adjust(left=0.20, right=0.74, top=0.96, bottom=0.10)

ax.axvspan(-22, 20, color="#f4a261", alpha=0.08, zorder=0)
ax.axvspan(20, 63, color="#2a9d8f", alpha=0.08, zorder=0)
ax.text(-11.5, 0.700, "post-encoding scoring\n(0\\% savings by construction)",
        fontsize=8.5, color="#c1440e", va="bottom", ha="left", style="italic")
ax.text(61, 0.700, "pixel-level scoring\n(encoder-compute savings possible)",
        fontsize=8.5, color="#1b6e64", va="bottom", ha="right", style="italic")

# --- greedy label stacking so tightly-clustered points get non-overlapping labels ---
def stacked_label_ys(items, min_gap):
    # items: list of (y, key) sorted ascending by y; returns dict key -> label_y
    ys = sorted(items, key=lambda t: t[0])
    out = {}
    prev = None
    for y, key in ys:
        ly = y if prev is None else max(y, prev + min_gap)
        out[key] = ly
        prev = ly
    return out

left_cluster = [(y, lbl) for (lbl, x, y, s, h, p) in points if not p]
right_cluster = [(y, lbl) for (lbl, x, y, s, h, p) in points if p]
left_ys = stacked_label_ys(left_cluster, 0.011)
right_ys = stacked_label_ys(right_cluster, 0.011)

LEFT_LABEL_X, RIGHT_LABEL_X = -20.5, 62.5

for label, x, y, std, headline, pixel in points:
    color = "#1b6e64" if pixel else "#c1440e"
    label_x = RIGHT_LABEL_X if pixel else LEFT_LABEL_X
    label_y = (right_ys if pixel else left_ys)[label]
    ha = "left" if pixel else "right"
    ms = 9 if headline else 5.5
    fw = "bold" if headline else "normal"
    fs = 9.5 if headline else 8
    alpha = 1.0 if headline else 0.6

    ax.errorbar(x, y, yerr=std, fmt="o", color=color, ecolor=color,
                elinewidth=1.1 if headline else 0.7, capsize=3 if headline else 2,
                markersize=ms, markeredgecolor="black" if headline else "none",
                markeredgewidth=0.8, alpha=alpha, zorder=5 if headline else 3)
    ax.plot([x, label_x], [y, label_y], color=color, alpha=0.35, linewidth=0.6, zorder=2)
    ax.annotate(label, (label_x, label_y), fontsize=fs, fontweight=fw,
                ha=ha, va="center", color=color, zorder=6)

ax.set_xlabel("ViT-encoder compute savings vs.\\ Uniform (\\%)")
ax.set_ylabel("macro-Jaccard (Cholec80, BiGRU, main setting)")
ax.set_xlim(-22, 63)
ax.set_ylim(0.695, 0.79)
ax.axvline(20, color="gray", linewidth=0.7, linestyle=":", zorder=1)
ax.grid(True, linewidth=0.4, alpha=0.4, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.savefig("/home/luning/CodeRepo/SurgKeyMAE/CMPBScorer/figures/frontier.pdf")
print("saved frontier.pdf")
