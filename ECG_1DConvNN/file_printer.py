import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pandas import Series

# file = open('ecgjune.txt')
file = open('newecg3.txt')
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

# DESI tune more
# arr = [(a-2400)/2600 for a in arr]

# baseline filter
# 0.001 <= p <= 0.1
# 10^2 ≤ λ ≤ 10^9
def baseline_als(y, lam=1, p=0.001, niter=5):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z



# Moving Average Filter
series = Series(arr)
# print(series)

# MASK 
#Define window size
w=2
#Define mask and store as an array
mask=np.ones((1,w))/w
mask=mask[0,:]
# convolve
convolved_data=np.convolve(series,mask,'same')

series=Series.to_frame(series)
series['convolved_data']=convolved_data

arr = series['convolved_data'].tolist()
# print(arr)

#Plot both original and smooth data

# plt.plot(series)
# plt.show()


# Double Der Normalize

minval = min(arr)
maxval = max(arr)
arr = [2*(val-minval)/(maxval-minval)-1 for val in arr]
# arr = []
print(arr)

# PLOT
x =  [i for i in range(len(arr))]

y = arr
plt.plot(x, y)

# baseline start
# arr = baseline_als(arr)
# x =  [i for i in range(len(arr))]

# y = arr
# plt.plot(x, y)
# baseline end

plt.show()
