import pandas as pn
from sklearn.model_selection import train_test_split
import numpy as np


def read_input(split_percent, test_percent, data_path, novel_class):
    data = pn.read_csv(data_path, header=None)
    features = len(data.columns) - 1

    # # preprocess data
    data = data.fillna((data.mean(axis=0)))

    # taking a portion of the full dataset that has no nobel class
    data_r, data = train_test_split(data, test_size=split_percent)
    t={}
    t=np.array_split()
    # separate novel classes
    n_class = []
    n_class1=[]
    for i in novel_class:
        n = data.loc[data[features] == i]
        n_class.append(n)
        data = data.loc[data[features] != i]

        n1 = data_r.loc[data_r[features] == i]
        n_class1.append(n1)
        data_r = data_r.loc[data_r[features] != i]
    n_class = pn.concat(n_class)
    n_class1 = pn.concat(n_class1)


    x_trn, x_tst = train_test_split(data, test_size=test_percent)
    train = pn.DataFrame(x_trn)
    train.to_csv('../datasets/data/train.csv', header=None, index=False, encoding="utf-8")
    test = pn.DataFrame(x_tst)
    s_n = len(test)
    test = test.append(n_class)
    e_n = len(test)
    test.to_csv('../datasets/data/test.csv', header=None, index=False, encoding="utf-8")
    X_train = train.iloc[:, 0:features]
    y_train = train.iloc[:, features]
    X_test = test.iloc[:, 0:features]
    y_test = test.iloc[:, features]
    ##process: finding nobel class starting and ending indexs
    n_index = [0 for i in range(0, 6)]
    n_index[0] = s_n
    n_index[1] = e_n - 1
    # 2nd test phase
    s_n=len(data_r)
    data_r=data_r.append(n_class1)
    e_n=len(data_r)
    n_index[2] = s_n
    n_index[3] = e_n - 1
    X_retest = data_r.iloc[:, 0:features]
    y_retest = data_r.iloc[:, features]
    return X_train, X_test, y_train, y_test, features, n_index,X_retest,y_retest
