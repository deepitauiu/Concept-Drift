from Modules.repo import reader_2 as reader
from Modules.repo import helper5 as helper
from Modules.repo import models
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
from collections import Counter
import math
import graphviz
from sklearn import tree
import numpy as np
import pandas as pn
from sklearn.model_selection import train_test_split

# #process:read data
dot_data = StringIO()

# data_path = "../datasets/soya_bean/soya_bean.csv"
# novel_class = ['charcoal-rot','diaporthe-stem-canker']
data_path = "../datasets/image_seg/image_seg.csv"
novel_class = ['SKY','CEMENT']
##**comment:novel_index is begining and ending index of novel class

# #**comment:percentage of (novel data)
test_size = 200
##**comment:percentage of Full data
train_size = 800
X_train, X_test, y_train, y_test, num_of_feature, novel_index = reader.read_input(train_size,
                                                                                  test_size,
                                                                                  data_path,
                                                                                  novel_class)
# novel data label
new_label = []
# existing class label where novel data falls in
temp = []
n_i = 2
for r in range(0, 4):

    # #process: apply models
    clf = models.decision_tree()
    clf = clf.fit(X_train, y_train)

    # #process: accuracy find
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy", accuracy)

    # #process: train data

    # #process: train data predicted label
    train_key = clf.apply(X_train)

    # #process: train data mapping
    classes = (helper.unique(train_key))
    print("Train leaf classes")
    print(classes)
    train_map = helper.hash_map(X_train, train_key, classes)

    # #process: finding mean for each feature
    train_mean = helper.find_mean(num_of_feature, train_map, classes)
    print('Train Mean')
    print(train_mean)

    # #process: finding max and min distance for each class

    # #***comment: for maxdistance consider index=0 and mindistance consider index=1
    train_distance = helper.find_distance(num_of_feature, train_map, train_mean, classes)
    print('Distance')
    print(train_distance)

    # #process: number of instance occurrences for each class
    print("Occurrences")
    print(Counter(train_key))

    # #process: train end


    # #process: test start

    # #process:test predicted value
    test_key = clf.apply(X_test)

    # #process: mapping test data
    test_map = helper.hash_map(X_test, test_key, classes)
    test_classes = (helper.unique(test_key))
    print("test leaf classes")
    print(test_classes)

    # process: finding test data leaf classes indexs according to train data indexs
    test_index = [0 for i in range(0, len(test_key))]
    j = 0
    for i in test_key:
        test_index[j] = classes.index(i)
        j += 1
    # #process: classify through distance measurement(max,min)

    # #**comment: pred is predictions of correct incorrect classify array
    pred = []
    count = 0
    c = 0
    index = 0
    test_data = helper.thash_map(X_test, test_key)
    # test data labels
    label = []
    label = y_test.tolist()

    for i in range(0, 1):
        for x in test_data[i]:
            dis = 0
            index = test_index[count]
            for j in range(0, num_of_feature):
                dis += math.fabs(train_mean[index][j] - x[j])
            # dis = math.fabs(dis)
            if dis >= 0 and dis <= train_distance[index][0]:
                pred.append(1)
                # print('correct')
            else:
                pred.append(0)
                # print('incorrect')
                p = y_pred[count]
                flag = 0
                c = 0
                for t_l in temp:
                    if t_l == p:
                        flag = 1
                        break
                    c += 1
                if flag == 0:
                    temp.append(p)
                    label[count] = 'n' + str(c)
                else:
                    label[count] = 'n' + str(c)

                new_label.append(label[count])
                new_label = helper.unique(new_label)

                # #process: append novel data with new level
                f = open("../datasets/data/train.csv", "a+")
                for n in x:
                    f.write("%.2f," % n)

                f.write("%s" % label[count])
                f.write("\n")
            count += 1
    print("temp")
    print(temp)

    # #process: testEnd

    # #Comment:outputs
    Xt = len(X_train)

    # #process:exsisting class find
    rc_exsists = y_test[0:novel_index[0]]
    # #**comment c for index and Fe count of miss classifying exsisting class
    c = 0
    Fe = 0
    Fp = 0
    for i in rc_exsists:
        if i != y_pred[c]:
            Fe += 1
        elif (pred[c] == 0):
            Fp += 1
        c += 1

    # #process:assumptions according to paper:
    N = len(X_test)
    Nc = len(X_test) - novel_index[0]

    # #For first test data phase

    if r == 0:
        # #process: accuracy of novel class
        nc_accuracy = 100 * (pred[novel_index[0]:novel_index[1]+1].count(0)) / Nc
        Nd = pred[novel_index[0]:novel_index[1]+1].count(0)

        # Fp =abs(Fe- pred[0:novel_index[0] - 1].count(0))
        Fn = pred[novel_index[0]:novel_index[1]+1].count(1)
    # #process: For second test data phase
    else:
        Nd = 0
        Fn = 0
        # #process: accuracy of novel class
        for i in range(novel_index[0], novel_index[1]+1):
            prd_nc = y_pred[i]
            if prd_nc in new_label:
                Nd += 1
            elif pred[i] == 0:
                Nd += 1
            else:
                Fn += 1

        nc_accuracy = 100 * Nd / Nc

    Mnew = (Fn * 100) / Nc
    Fnew = (Fp * 100) / (N - Nc)
    ERR = ((Fp + Fn + Fe) * 100) / N
    # #Process: overall accuracy
    total_accuracy = (Nd+c-(Fe+Fp)) * 100 / N
    helper.print_data(novel_index, X_test, pred, Xt, N, Nc, Fp, Nd, nc_accuracy, Fn, Fe, Mnew, Fnew, ERR,
                      total_accuracy)

    # #process: printing results
    if (nc_accuracy >= 0.00):
        helper.results(Xt, N, Nc, Nd, nc_accuracy, total_accuracy, Mnew, Fnew, ERR)

    # #process: retrain our model
    X_train, y_train = helper.retrain(num_of_feature)
    X_test, y_test = helper.retest(num_of_feature, r + 1)

    novel_index[0] = novel_index[n_i]
    novel_index[1] = novel_index[n_i + 1]
    n_i += 2

# #process: printing data
f = open("../datasets/reports/report.csv", "a+")
f.write("\n")
