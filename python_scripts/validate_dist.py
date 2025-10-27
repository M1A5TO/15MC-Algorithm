import numpy as np

d = np.load("test_validate_csr.npz")
indptr, indices, weights, lonlat = d["indptr"], d["indices"], d["weights"], d["lonlat"]
N = len(indptr)-1
u = np.repeat(np.arange(N, dtype=np.int32), indptr[1:]-indptr[:-1])
v = indices

def haversine_m(a, b):
    R=6371000.0
    dlon=np.radians(lonlat[b,0]-lonlat[a,0])
    dlat=np.radians(lonlat[b,1]-lonlat[a,1])
    x=np.sin(dlat/2)**2 + np.cos(np.radians(lonlat[a,1]))*np.cos(np.radians(lonlat[b,1]))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(x))

h = haversine_m(u, v)
ratio = weights / np.maximum(h, 1e-6)

print("median ratio:", np.median(ratio))
print("p90 ratio:", np.percentile(ratio, 90))
print("bad edges (>2x):", (ratio>2.0).sum(), " / ", len(ratio))
