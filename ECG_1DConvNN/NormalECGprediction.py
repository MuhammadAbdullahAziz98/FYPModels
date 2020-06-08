
# this code use CNN model to predict normal ecg from dataset

# Note: Since this file uses dataset,
#  to run this file you need dataset folder in current directory, available at https://physionet.org/content/mitdb/1.0.0/

# Note : it first plots ECG after crossing the matplotlib graph sigmoid probabilities prediction is shown on terminal
# closer to zero means normal


import matplotlib.pyplot as plt
import numpy as np
import wfdb
from keras.models import load_model
import pandas as pd


data_path = 'mit-bih-arrhythmia-database-1.0.0/'
pts=['100']

df = pd.DataFrame()

for pt in pts:
    file = data_path +pt
    annotation = wfdb.rdann(file,'atr')
    sym = annotation.symbol

    values, counts = np.unique(sym, return_counts=True)
    df_sub = pd.DataFrame({'sym':values,'val':counts,'pt':[pt]*len(counts)})
    df = pd.concat([df,df_sub],axis =0)

abnormal = ['L','R','V','/','A','f','F','g','a','E','J','e','S']
nonbeat = ['[','!',']','x','(',')','p','t','u','`','\'','^','|','~','+','s','T','*','D','=','"','@','Q','?']

df['cat'] = -1
df.loc[df.sym=='N','cat'] = 0
df.loc[df.sym.isin(abnormal),'cat'] = 1

def load_ecg(file):
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file,'atr')
    p_signal = record.p_signal

    assert record.fs == 360,'sample freq not 360'

    
    atr_sym = annotation.symbol
    
    atr_sample = annotation.sample
    return p_signal, atr_sym, atr_sample

p_signal, atr_sym, atr_sample = load_ecg(data_path+"100")
# print(atr_sym[3])

def make_dataset(pts, num_sec, fs):

    num_cols = 2*num_sec*fs
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    sym_all = []
    max_rows= []

    for pt in pts:
        file = data_path + pt
        p_signal, atr_sym, atr_sample = load_ecg(file)
    
        p_signal = p_signal[:,0]
        
        df_ann = pd.DataFrame({'atr_sym':atr_sym,'atr_sample':atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal+['N'])]

        num_rows = len(df_ann)
        X = np.zeros((num_rows,num_cols))
        Y = np.zeros((num_rows,1))
        max_row = 0
        for atr_sample, atr_sym in zip(df_ann.atr_sample.values,df_ann.atr_sym.values):
            left = max([0,(atr_sample - num_sec*fs) ])
            right = min([len(p_signal),(atr_sample + num_sec*fs) ])
            x = p_signal[left: right]
            if len(x) == num_cols:
                X[max_row,:] = x
                Y[max_row,:] = int(atr_sym in abnormal)
                sym_all.append(atr_sym)
                max_row += 1
        X = X[:max_row,:]
        Y = Y[:max_row,:]
        max_rows.append(max_row)
        X_all = np.append(X_all,X,axis = 0)
        Y_all = np.append(Y_all,Y,axis = 0)
    # drop the first zero row
    X_all = X_all[1:,:]
    Y_all = Y_all[1:,:]
    # check sizes make sense
    assert np.sum(max_rows) == X_all.shape[0], 'number of rows messed up'
    assert Y_all.shape[0] == X_all.shape[0], 'number of rows messed up'
    assert Y_all.shape[0] == len(sym_all), 'number of rows messed up'
    return X_all, Y_all, sym_all



num_sec = 3
fs = 360
X_all, Y_all, sym_all = make_dataset(pts, num_sec, fs)



arr02 = X_all[0].tolist()


p_signal = p_signal[:,0]

arr = p_signal
# print(len(arr))
# print(len(atr_sym))


app_len=0
if len(arr)<2160:
    app_len = 2160 - len(arr) 
    for i in range(app_len):
        arr.append(arr[i])

# +600 for normal
elif len(arr)>2160:
    arr2 = [arr[a] for a in range(0,2160)]
    arr = arr2

x =  [i for i in range(len(arr))]
y = arr
# plt.plot(x, y)
# plt.plot(x,X_all[10].tolist())
plt.plot(x,X_all[3].tolist())

plt.show()


#  ----------prediction --------

model = load_model('models/modelcnn.h5')

# print(sym_all)
# print(sym_all[0])


arr = X_all[0].tolist()
arr = np.array(arr)
arr = np.reshape(arr,(1,2160,1))

Y_Normal_Dataset = model.predict_proba(arr,verbose = 1)

print(Y_Normal_Dataset)



