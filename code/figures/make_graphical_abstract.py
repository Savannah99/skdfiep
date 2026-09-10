"""Graphical abstract: a simplified version of the accuracy/compute frontier
plot, showing only the three headline methods discussed in the paper plus
Uniform as a reference point, with larger fonts suited to a small thumbnail.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 13})

points = [
    ("Uniform\n(all frames)",  0.0, 0.7147, False, False, (0, 14)),
    ("Fixed-position",        50.0, 0.7508, True,  True,  (-18, -20)),
    ("MobileNet-TS+\n(ours)", 48.5, 0.7543, True,  True,  (-8, 20)),
    ("SCSampler\n(ceiling)",   0.0, 0.7565, True,  False, (0, 16)),
]

fig, ax = plt.subplots(figsize=(6.4, 5.0))
fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.15)

ax.axvspan(-8, 20, color="#f4a261", alpha=0.10, zorder=0)
ax.axvspan(20, 63, color="#2a9d8f", alpha=0.10, zorder=0)

for label, x, y, headline, pixel, xytext in points:
    color = "#1b6e64" if pixel else "#c1440e"
    ms = 16 if headline else 9
    ax.plot(x, y, "o", color=color, markersize=ms,
            markeredgecolor="black" if headline else "none", markeredgewidth=1.0, zorder=5)
    va = "top" if xytext[1] < 0 else "bottom"
    ax.annotate(label, (x, y), textcoords="offset points", xytext=xytext,
                fontsize=12.5 if headline else 10.5,
                fontweight="bold" if headline else "normal",
                ha="center", va=va, color=color, zorder=6)

ax.annotate("", xy=(48.5, 0.7543), xytext=(50.0, 0.7508),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, linestyle="dashed"))

ax.set_xlabel("ViT-encoder compute savings (\\%)", fontsize=13)
ax.set_ylabel("macro-Jaccard", fontsize=13)
ax.set_title("Frame sampling for surgical phase recognition:\naccuracy vs.\\ compute", fontsize=13.5, pad=10)
ax.set_xlim(-8, 63)
ax.set_ylim(0.705, 0.775)
ax.grid(True, linewidth=0.4, alpha=0.4, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.savefig("/home/luning/CodeRepo/SurgKeyMAE/CMPBScorer/figures/graphical_abstract.pdf")
print("saved graphical_abstract.pdf")
