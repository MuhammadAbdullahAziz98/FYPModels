

# this code utilizes the CNN model and give prediction on Abdullah Aziz ecg
# Note : it first plots ECG after crossing the matplotlib graph sigmoid probabilities prediction is shown on terminal
# closer to zero means normal 

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from pandas import Series
from scipy import sparse
from scipy.sparse.linalg import spsolve


file = open('ecgnotrobot.txt')




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

# noise removal
def m_Average(arr):
        # MASK 
    #Define window size
    series = Series(arr)
    w=2
    #Define mask and store as an array
    mask=np.ones((1,w))/w
    mask=mask[0,:]
    # convolve
    convolved_data=np.convolve(series,mask,'same')

    series=Series.to_frame(series)
    series['convolved_data']=convolved_data
    arr = series['convolved_data'].tolist()
    return arr
    


# noise removal
arr = m_Average(arr)
arr[0] = arr[1]
# baseline
#arr = baseline_als(arr)

# #Double Der Normalize
minval = min(arr)
maxval = max(arr)
arr = [2*(val-minval)/(maxval-minval)-1 for val in arr]



x =  [i for i in range(len(arr))]

y = arr
plt.plot(x, y)
plt.show()

# ---------- prediction------------


model = load_model('models/modelcnn.h5')

arr = np.array(arr)
arr = np.reshape(arr,(1,2160,1))

Y_abdullah_ECG = model.predict_proba(arr,verbose = 1)

print(Y_abdullah_ECG)

