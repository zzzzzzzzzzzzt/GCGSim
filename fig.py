import matplotlib.pyplot as plt
import numpy as np

datasetname = ("AIDS700nef", "LINUX", "IMDBMulti")
penguin_means = {
    'PKGD-GMN':               (1.162, 0.066, 0.550),
    'PKGD-GMN(w/o PKGD)':      (1.375, 0.113, 0.575),
    'PKGD-GMN-ex':            (1.390, 0.067, 0.449),
    'PKGD-GMN(w/o PKGD)-ex':   (6.050, 0.683, 1.438),
}
colors = ['#0070C0', '#40B0FF', '#92D050', '#BEE396']
x = np.arange(len(datasetname))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(5, 2.5), layout='constrained')

for i, (attribute, measurement) in enumerate(penguin_means.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, align='edge', color=colors[i], zorder=3)
    ax.bar_label(rects, padding=1, fmt='%.3f', fontsize=6)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE(1e-3)')
# ax.set_title('Penguin attributes by species')
ax.set_xticks(x+2*width, datasetname)
ax.legend()
ax.set_ylim(0, 7)
plt.grid(axis='y', color='0.8', zorder=0)

fig.savefig('fig5.pdf', bbox_inches='tight')