import re
from utils.visualization import plot_ablation_summary

ROW = r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
def val_accs(log):
    return [float(r[4]) for r in re.findall(ROW, open(log).read(), re.M)]

results = {
    # In-lab runs (CPU, 5 epochs), values from the terminal output
    "1. Base LeNet-5":        {"val_accs": [46.28, 49.97, 49.83, 52.33, 53.16]},
    "2. Modernized LeNet":    {"val_accs": [52.06, 59.44, 62.45, 64.73, 64.51]},
    "3. AlexNet-64 (No Norm)": {"val_accs": val_accs("logs/alexnet_none.txt")},
    "4. AlexNet-64 + LRN":    {"val_accs": val_accs("logs/alexnet_lrn.txt")},
    "5. AlexNet-64 + BN":     {"val_accs": val_accs("logs/alexnet_bn.txt")},
    "6. AlexNet-64 + BN + Aug": {"val_accs": val_accs("logs/alexnet_with_aug.txt")},
    "7. Zoo AlexNet":         {"val_accs": val_accs("logs/zoo_alexnet.txt")},
}
for k, v in results.items():
    print(f"{k:28s} {len(v['val_accs'])} epochs, best {max(v['val_accs']):.2f}")
plot_ablation_summary(results, save_path="./plots/ablation_summary.png")
