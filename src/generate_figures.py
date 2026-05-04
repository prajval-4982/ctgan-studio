"""Generate publication-quality figures from benchmark results."""
import sys, os, io, warnings
warnings.filterwarnings("ignore")

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "Research_paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Actual experimental results
epsilons = [0.5, 1.0, 5.0, 10.0]

# F1 scores per dataset per epsilon
f1 = {
    "ILPD":  [0.1448, 0.5664, 0.5275, 0.1448],
    "Pima":  [0.5293, 0.5145, 0.2352, 0.5031],
    "Adult": [0.0853, 0.2503, 0.7645, 0.6713],
}

# MIA accuracy per dataset per epsilon (as percentage)
mia = {
    "ILPD":  [61.27, 62.86, 58.73, 62.54],
    "Pima":  [61.20, 61.93, 64.34, 63.86],
    "Adult": [60.67, 60.89, 59.30, 60.33],
}

# PU scores
pu = {
    "ILPD":  [0.2108, 0.4486, 0.4631, 0.2089],
    "Pima":  [0.4478, 0.4376, 0.2834, 0.4206],
    "Adult": [0.1402, 0.3052, 0.5312, 0.4987],
}

# Baselines
baselines = {
    "ILPD":  {"vanilla_f1": 0.5418, "smote_f1": 0.5811, "real_f1": 0.5538,
              "vanilla_mia": 65.40, "smote_mia": 68.57,
              "vanilla_pu": 0.4223, "smote_pu": 0.4080},
    "Pima":  {"vanilla_f1": 0.6231, "smote_f1": 0.7388, "real_f1": 0.7213,
              "vanilla_mia": 64.34, "smote_mia": 74.22,
              "vanilla_pu": 0.4536, "smote_pu": 0.3822},
    "Adult": {"vanilla_f1": 0.7000, "smote_f1": 0.8335, "real_f1": 0.8542,
              "vanilla_mia": 61.70, "smote_mia": 68.22,
              "vanilla_pu": 0.4951, "smote_pu": 0.4602},
}

# ----------------------------------------------------------------
# Figure 1: Privacy-Utility Trade-off Curves
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
colors_f1 = {'ILPD': '#2196F3', 'Pima': '#4CAF50', 'Adult': '#FF9800'}
colors_mia = {'ILPD': '#1565C0', 'Pima': '#2E7D32', 'Adult': '#E65100'}

for ax, ds in zip(axes, ["ILPD", "Pima", "Adult"]):
    ax.plot(epsilons, [v*100 for v in f1[ds]], 'o--', color=colors_f1[ds],
            linewidth=2, markersize=7, label='F1-Score (%)', zorder=5)
    ax.plot(epsilons, mia[ds], 's-', color=colors_mia[ds],
            linewidth=2, markersize=7, label='MIA Accuracy (%)', zorder=5)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.6, label='Random Guess (50%)')
    ax.axhline(y=baselines[ds]["real_f1"]*100, color=colors_f1[ds],
               linestyle=':', alpha=0.4, label=f'Real F1 ({baselines[ds]["real_f1"]*100:.1f}%)')
    ax.set_xlabel(r'Privacy Budget ($\varepsilon$)', fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title(ds, fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Privacy-Utility Trade-off Across Datasets', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_tradeoff.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(FIG_DIR, "fig_tradeoff.png"), bbox_inches='tight', dpi=300)
print("[OK] fig_tradeoff.pdf/png saved")

# ----------------------------------------------------------------
# Figure 2: PU-Score Grouped Bar Chart
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

methods = ['Vanilla\nCTGAN', 'SMOTE',
           r'DP-CTGAN'+'\n'+r'$\varepsilon$=0.5',
           r'DP-CTGAN'+'\n'+r'$\varepsilon$=1.0',
           r'DP-CTGAN'+'\n'+r'$\varepsilon$=5.0',
           r'DP-CTGAN'+'\n'+r'$\varepsilon$=10.0']

ilpd_scores = [baselines["ILPD"]["vanilla_pu"], baselines["ILPD"]["smote_pu"]] + pu["ILPD"]
pima_scores = [baselines["Pima"]["vanilla_pu"], baselines["Pima"]["smote_pu"]] + pu["Pima"]
adult_scores = [baselines["Adult"]["vanilla_pu"], baselines["Adult"]["smote_pu"]] + pu["Adult"]

x = np.arange(len(methods))
w = 0.25

bars1 = ax.bar(x - w, ilpd_scores, w, label='ILPD', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x, pima_scores, w, label='Pima', color='#4CAF50', alpha=0.85)
bars3 = ax.bar(x + w, adult_scores, w, label='Adult', color='#FF9800', alpha=0.85)

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7)

# Highlight best
best_idx = 3  # DP-CTGAN e=1.0 for ILPD; but best varies per dataset
# Just highlight the e=5.0 Adult bar which is the highest overall
ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='Threshold (0.50)')

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('PU-Score', fontsize=12)
ax.set_title('Privacy-Utility Score Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0, 0.7)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_puscore.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(FIG_DIR, "fig_puscore.png"), bbox_inches='tight', dpi=300)
print("[OK] fig_puscore.pdf/png saved")

print("\nAll figures generated in:", FIG_DIR)
