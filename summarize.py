import re, glob, os
from utils.seed import format_mean_std

ROW = r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
rows = {}
for f in sorted(glob.glob("logs/*.txt")):
    t = open(f).read()
    m = re.search(r"Best Val Accuracy: ([\d.]+)% \| Best Val Macro F1: ([\d.]+)%", t)
    c = re.search(r"Training completed in ([\d.]+)s", t)
    ep = re.findall(ROW, t, re.M)
    name = os.path.basename(f)[:-4]
    if not (m and c and ep):
        print(f"{name}: INCOMPLETE"); continue
    n, last = len(ep), ep[-1]
    rows[name] = (float(m[1]), float(m[2]))
    print(f"{name:26s} best {float(m[1]):6.2f} f1 {float(m[2]):6.2f} | final train {float(last[2]):6.2f} val {float(last[4]):6.2f} | {n} ep {float(c[1])/n:6.1f} s/ep")

print("\n--- Mean +/- Std over seeds 42, 123, 2026 ---")
groups = {"none": ["alexnet_none", "alexnet_none_s123", "alexnet_none_s2026"],
          "LRN": ["alexnet_lrn", "alexnet_lrn_s123", "alexnet_lrn_s2026"],
          "BN (lr1e-3, bs128)": ["alexnet_bn", "alexnet_bn_s123", "alexnet_bn_s2026"]}
for k in ["lr1e-4", "lr3e-3", "bs64", "bs256"]:
    groups[k] = [f"abl_{k}_s{s}" for s in (42, 123, 2026)]
for g, names in groups.items():
    a = [rows[x][0] for x in names if x in rows]; b = [rows[x][1] for x in names if x in rows]
    if len(a) >= 2:
        print(f"{g:20s} n={len(a)}  acc {format_mean_std(a)}  f1 {format_mean_std(b)}")

print("\n--- Tiny ImageNet trajectories (train acc / val acc / val loss) ---")
for x in ["tiny_alexnet_bn_aug", "tiny_alexnet_bn", "tiny_alexnet_none_lr1e-4"]:
    ep = re.findall(ROW, open(f"logs/{x}.txt").read(), re.M)
    print(x)
    for e in ep:
        print(f"  ep {e[0]:>2}: {e[2]:>6} / {e[4]:>6} / {e[3]}")
