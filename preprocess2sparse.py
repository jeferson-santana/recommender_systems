from __future__ import print_function, division
from builtins import range, input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

# load in the data
df = pd.read_csv('C:/Users/Jeferson Santana/OneDrive/Documentos/recommender_course/data/edited_rating.csv')

N = df.userId.max() + 1 # number of user
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

A = lil_matrix((N, M))
print("Calling: updated_train")
count = 0
def update_train(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/cutoff))

    i = int(row.userId)
    j = int(row.movie_idx)
    A[i, j] = row.rating
df_train.apply(update_train, axis=1)

# mask, to tell us which entries exist and which do not
A = A.tocsr()
mask = (A > 0)
save_npz("Atrain.npz", A)

A_test = lil_matrix((N, M))
print("Calling: updated_test")
count = 0
def update_test(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/cutoff))

    i = int(row.userId)
    j = int(row.movie_idx)
    A[i, j] = row.rating
df_test.apply(update_test, axis=1)

# mask, to tell us which entries exist and which do not
A_test = A_test.tocsr()
mask = (A_test > 0)
save_npz("Atest.npz", A_test)