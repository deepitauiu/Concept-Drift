import pandas as pn
from sklearn.model_selection import train_test_split
import numpy as np


def read_input(train_size, test_size, data_path, novel_class):
    data = pn.read_csv(data_path, header=None)
    features = len(data.columns) - 1
    # novel data index
    n_index = [0 for i in range(0, 20)]

    # # preprocess data
    data = data.fillna((data.mean(axis=0)))
    # Prepare test phase
    c=0
    for i in range(0,6):
        n_class = []
        x = data.sample(n=test_size, replace=False)
        for j in novel_class:
            n = x.loc[x[features] == j]
            n_class.append(n)
            x = x.loc[x[features] != j]
        n_class = pn.concat(n_class)
        test = pn.DataFrame(x)
        s_n = len(test)
        test = test.append(n_class)
        e_n = len(test)
        test.to_csv('../datasets/data/test' + str(i) + '.csv', header=None, index=False, encoding="utf-8")
        n_index[c] = s_n
        n_index[c + 1] = e_n - 1
        c+=2
    x = data.sample(n=train_size, replace=False)
    for i in novel_class:
        x = x.loc[x[features] != i]

    train = pn.DataFrame(x)
    train.to_csv('../datasets/data/train.csv', header=None, index=False, encoding="utf-8")
    t = pn.read_csv('../datasets/data/test0.csv', header=None)
    X_train = train.iloc[:, 0:features]
    y_train = train.iloc[:, features]
    X_test = t.iloc[:, 0:features]
    y_test = t.iloc[:, features]

    return X_train, X_test, y_train, y_test, features, n_index
