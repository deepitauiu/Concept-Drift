import math
import pandas as pn
import numpy as np


# hashmap
def hash_map(params, pred_class, classes, y_train):
    hashMap = {}
    class_label = [0 for j in range(0, len(classes))]
    count = 0
    # cl_len = len(params.columns)
    # cl_count = 0
    y_train = y_train.tolist()
    for i in pred_class:
        class_label[classes.index(i)] = y_train[count]
        hashMap.setdefault(classes.index(i), []).append(params.iloc[count])
        count += 1
    return hashMap, class_label


# finding mean for each feature
def find_mean(num_of_feature, map, classes):
    l = len(classes)
    mean = [[0 for i in range(0, num_of_feature)] for j in range(0, l)]
    # total mean
    for i in range(0, l):
        for x in map[i]:
            for j in range(0, num_of_feature):
                mean[i][j] += x[j]

    # average of feature mean
    for i in range(0, l):
        for j in range(0, num_of_feature):
            mean[i][j] /= len(map[i])
            mean[i][j] = float("{0:.2f}".format(mean[i][j]))

    return mean


# finding distance
# for maxdistance consider index=0 and mindistance consider index=1
def find_distance(num_of_feature, map, mean, classes, data_classes, new_label):
    l = len(classes)
    flag = 0
    # # among classes lowest maximum distance
    m_dis = 100.0
    f = open("../datasets/data/train1.csv", "a+")
    distance = [[0 for i in range(0, 2)] for j in range(0, l)]
    for i in range(0, l):
        distance[i][1] = 100
        distance[i][0] = 0.0
        avg = 0.0

        for x in map[i]:
            dis = 0
            for j in range(0, num_of_feature):
                dis += math.fabs(mean[i][j] - x[j])
                # dis = math.fabs(dis)
            dis = float("{0:.2f}".format(dis))
            if (dis > distance[i][0]):
                distance[i][0] = dis
            if (dis < distance[i][1]):
                distance[i][1] = dis
            avg += dis
        avg = float("{0:.2f}".format(avg / len(map[i])))
        # # calculate standard deviation
        sum = 0
        std = 0
        for st in map[i]:
            dis2 = 0
            for j in range(0, num_of_feature):
                dis2 += math.fabs(mean[i][j] - st[j])
            dis2 = float("{0:.2f}".format(dis2))
            sum += math.pow(dis2 - avg, 2)
        std = math.sqrt(sum / len(map[i]))
        # print("std", std)
        # #delete instance in standard deviation
        for instance in map[i]:
            dis1 = 0
            for j in range(0, num_of_feature):
                dis1 += math.fabs(mean[i][j] - instance[j])
                # dis = math.fabs(dis)
            dis1 = float("{0:.2f}".format(dis1))
            # if data_classes[i] in new_label:
            #     for j in range(0, num_of_feature):
            #         f.write("%0.2f," % instance[j])
            #     f.write("%s" % data_classes[i])
            #     f.write("\n")
            if dis1 < avg - (std/2) or dis1 > avg + (std/2):
                for j in range(0, num_of_feature):
                    f.write("%0.2f," % instance[j])
                f.write("%s" % data_classes[i])
                f.write("\n")

        if (distance[i][0] < m_dis and distance[i][0] > 0):
            m_dis = distance[i][0]
    for i in range(0, l):
        if len(map[i]) == 1:
            distance[i][0] = m_dis
    # print(distance)
    return distance


def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list


# test hash map
def thash_map(params, pred_class):
    hashMap = {}
    count = 0
    # cl_len = len(params.columns)
    # cl_count = 0
    for i in pred_class:
        hashMap.setdefault(0, []).append(params.iloc[count])
        count += 1
    return hashMap


def retrain(num_of_feature):
    re_train = pn.read_csv("../datasets/data/train1.csv", header=None)
    X_train = re_train.iloc[:, 0:num_of_feature]
    y_train = re_train.iloc[:, num_of_feature]
    return X_train, y_train


def retest(num_of_feature, r):
    re_test = pn.read_csv('../datasets/data/test' + str(r) + '.csv', header=None)
    X_test = re_test.iloc[:, 0:num_of_feature]
    y_test = re_test.iloc[:, num_of_feature]
    return X_test, y_test


# #process: results write on report file
def results(Xt, N, Nc, Nd, nc_accuracy, total_accuracy, Mnew, Fnew, ERR):
    # #process: printing data
    f = open("../datasets/reports/report.csv", "a+")
    # f.write("%d,%d,%d,%d,%f,%f,%f,%f" %Xt %N %Nc %Nd %nc_accuracy %Mnew %Fnew %ERR )
    # Train_data
    f.write("%d," % Xt)
    # Test_data
    f.write("%d," % N)
    # Total novel data
    f.write("%d," % Nc)
    # Detected number of novel data
    f.write("%d," % Nd)
    # Novel data detection accuracy
    f.write("%f," % nc_accuracy)
    # overall accuracy
    f.write("%f," % total_accuracy)
    f.write("%f," % Mnew)
    f.write("%f," % Fnew)
    f.write("%f," % ERR)
    f.write("\n")


def print_data(novel_index, X_test, pred, Xt, N, Nc, Fp, Nd, nc_accuracy, Fn, Fe, Mnew, Fnew, ERR, total_accuracy):
    # #process: prediction check in range
    print("Novel class start and end index number : ", novel_index)
    print("prediction of novel classes among", len(X_test) - novel_index[0], "novel instances:")
    print(pred[novel_index[0] :novel_index[1]+1])

    print("Total train data: ", Xt)
    print("Total miss classify among ", len(X_test), "test instances:")
    print(pred.count(0))

    print("Total instances in the data stream, N = ", N)
    print("Total novel class instances in the data stream, Nc= ", Nc)
    print("Total existing class instances misclassified as novel classes, Fp= ", Fp)

    print("Total novel classes detect among", len(X_test) - novel_index[0], "novel instances:")
    print(Nd)

    print("Accuracy of novel class detection:")
    print(nc_accuracy)

    print("Total novel class instances misclassified as existing classes, Fn= ", Fn)
    print("Total existing class instances misclassified, Fe=", Fe)
    print("% of novel class instances misclassified as existing classes, Mnew= ", Mnew)
    print("% of existing class instances falsely identified as novel classes, Fnew= ", Fnew)
    print("Total misclassification error, ERR = ", ERR)
    # #Process: overall accuracy
    print("overall accuracy total_accuracy = ", total_accuracy)
