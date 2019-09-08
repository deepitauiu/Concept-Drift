import pandas as pn
from sklearn.model_selection import train_test_split


def read_input(split_percent, test_percent, data_path, novel_class):
    data = pn.read_csv(data_path, header=None)
    features = len(data.columns) - 1

    # # preprocess data
    data = data.fillna((data.mean(axis=0)))

    # taking a portion of the full dataset that has no nobel class
    data_r, data = train_test_split(data, test_size=split_percent)
    # separate novel classes
    n_class = []
    for i in novel_class:
        n=data.loc[data[features] == i]
        n_class.append(n)
        data = data.loc[data[features] != i]
    n_class=pn.concat(n_class)

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
    n_index = [0 for i in range(0, 2)]
    n_index[0] = s_n
    n_index[1] = e_n-1
    return X_train, X_test, y_train, y_test, features, n_index
