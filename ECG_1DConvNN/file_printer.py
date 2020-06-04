import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

file = open('ecgjune.txt')
arr = []

for f in file:
    f = f.strip()
    if f== '':
        continue
    arr.append(f)


arr = [int(a) for a in arr]


app_len=0
if len(arr)<2160:
    app_len = 2160 - len(arr) 
    for i in range(app_len):
        arr.append(arr[i])

elif len(arr)>2160:
    arr2 = [arr[a] for a in range(0,2160)]
    arr = arr2

# tune more
# arr = [(a-2400)/2600 for a in arr]

# baseline filter
# 0.001 <= p <= 0.1
# 10^2 ≤ λ ≤ 10^9
def baseline_als(y, lam=10, p=0.1, niter=3):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z


# arr = baseline_als(arr)

x =  [i for i in range(len(arr))]

y = arr
plt.plot(x, y)
plt.show()
