import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--csr", required=True)
args = ap.parse_args()

npz = np.load(args.csr)
indptr  = npz["indptr"]; indices = npz["indices"]; w = npz["weights"]; lonlat = npz["lonlat"]
N = indptr.shape[0] - 1; E = len(indices)

outdeg = indptr[1:] - indptr[:-1]
deg_mean = float(outdeg.mean())
zero_deg = int((outdeg == 0).sum())

print(f"N={N}, E={E}")
print(f"out-degree: mean={deg_mean:.2f}, max={outdeg.max()}, zero-outdeg={zero_deg} ({100*zero_deg/N:.2f}%)")
print(f"lon range=({lonlat[:,0].min():.4f}, {lonlat[:,0].max():.4f})  lat range=({lonlat[:,1].min():.4f}, {lonlat[:,1].max():.4f})")

# reciprocity (próbkowo)
sample = E
pairs = []
u = 0
for e in range(sample):
    while not (indptr[u] <= e < indptr[u+1]): u += 1
    pairs.append((u, int(indices[e])))
S = set(pairs)
rec = sum((v, u) in S for (u, v) in pairs)
print(f"reciprocity ~ {100*rec/len(pairs):.2f}%")