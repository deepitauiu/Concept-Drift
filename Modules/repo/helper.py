import math


# hashmap
def hash_map(params, pred_class, classes):
    hashMap = {}
    count = 0
    # cl_len = len(params.columns)
    # cl_count = 0
    for i in pred_class:
        hashMap.setdefault(classes.index(i), []).append(params.iloc[count])
        count += 1
    return hashMap


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
def find_distance(num_of_feature, map, mean, classes):
    l = len(classes)
    flag = 0
    # # among classes lowest maximum distance
    m_dis = 100.0
    distance = [[0 for i in range(0, 2)] for j in range(0, l)]
    for i in range(0, l):
        distance[i][1] = 100
        distance[i][0] = 0.0
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
        if (distance[i][0] < m_dis and distance[i][0]>0):
            m_dis = distance[i][0]
    for i in range(0, l):
        if (len(map[i]) == 1):
            distance[i][0]=m_dis

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