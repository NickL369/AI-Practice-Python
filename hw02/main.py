import numpy as np
import time

f = np.genfromtxt("hw02/data/iris.data", delimiter=",")
x = f[:, [0, 1, 2, 3]]

m1 = np.nanmean(x[:, 0])
m2 = np.nanmedian(x[:, 0])
m3 = np.nanstd(x[:, 0])
print(m1, m2, m3)

cells = x.size
idx = np.random.choice(cells, 20, replace=False)
x.ravel()[idx] = np.nan

nan_rows = np.where(np.isnan(x[:, 0]))[0]
print(nan_rows)

m_filter = (x[:, 2] > 1.5) & (x[:, 0] < 5.0)
subset = x[m_filter]

tmp_flat = x.flatten()
clean_list = list(map(lambda v: 0.0 if np.isnan(v) else v, tmp_flat))
x = np.array(clean_list).reshape(x.shape)

u, c = np.unique(x, return_counts=True)
print(np.stack((u, c), axis=1))

a, b = np.split(x, 2, axis=1)

a = a[np.argsort(a[:, 0])]

s_idx = np.argsort(b[:, 0])
s_idx_desc = s_idx[::-1]
b = b[s_idx_desc]

x = np.concatenate([a, b], axis=1)

vals, counts = np.unique(x, return_counts=True)
mode_idx = np.argmax(counts)
print(vals[mode_idx])

def transform(column):
    avg = np.mean(column)
    res = np.zeros_like(column)
    low_mask = column < avg
    res[low_mask] = column[low_mask] * 2
    high_mask = column >= avg
    res[high_mask] = column[high_mask] / 4
    return res

for _ in range(10_000):
    transform(x[:, 2])

t0 = time.perf_counter()
x[:, 2] = transform(x[:, 2])
t1 = time.perf_counter()

print("transform time ms =", (t1 - t0) * 1000)

print(x.shape)
print(len(subset))
