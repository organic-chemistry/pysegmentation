
import numpy as np
import pandas as pd
def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())


def load_data(root,bin_size):
    simu_data=[]
    with open(root,"r") as f:  #.fa
        l = f.readlines()
        l=l[1::2]
        simu_data=[]
        for sl in l[:]:
            simu_data.append(smooth(np.array(list(map(float,sl.strip().split()))),100)[::100])
    return simu_data

def load_data_ml(root,bin_size,read_size):
    noisy = load_data(root +".fa",bin_size)
    states =  load_data(root + '_states.fa',bin_size)

    X = []
    y=[]
    for i in range(len(noisy)):
        if len(noisy[i]) >= read_size:
            X.append(noisy[i][:read_size])
            y.append(np.array(states[i][:read_size],dtype=int))
    return X,y
