import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from timeit import default_timer as timer
import numpy as np

tic = timer()

df = pd.read_feather('train.feather').set_index('row_id')

FRAC = 0.50
df = df.sample(frac = FRAC)
print(f'{FRAC*100}%: ')
df.info()


int_columns = df.select_dtypes(int).columns 
df[int_columns] = df[int_columns].astype('int16')

float_columns = df.select_dtypes(float).columns 
df[float_columns] = df[float_columns].astype('float32')

print('Downcasted: ')
df.info()

X = df.drop(columns = ['target'])
y = df.target

result = pd.Series(mutual_info_regression(X = X, y = y), index = X.columns)

result.to_csv('mutual_info_reg.csv')

toc = timer()

print(f'Total elapsed time: {(toc - tic)/60} minutes')