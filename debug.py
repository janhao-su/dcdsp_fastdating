import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    f = open('output2.txt', 'w')
    print(x)
    print >> f, x
    pd.reset_option('display.max_rows')
    f.close()

data_df = pd.read_csv("speed_dating_train.csv", encoding="ISO-8859-1")
# take a quick look into Data
print(data_df.head())
print(data_df.describe())

# see NaN
print_full(data_df.isnull().sum())