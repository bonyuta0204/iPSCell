
# coding: utf-8

import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("/home/share/libraries/bdpy")
import bdpy
from bdpy.preproc import select_top
from bdpy.ml import add_bias, make_cvindex
from bdpy.stats import corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC




# list for sampling features
NUM_FEATURES = [i for i in range(1, 100)] + [i * 50 for i in range(2, 40)]



# ventricle: 0, article: 1
# choose 71 samples from vertricle and make test data and label data
def make_dataset(v_spec, a_spec):
    """
    choose 71 samples from vertricle and make test data and label data
    """
    N = a_spec.shape[0]
    choice = np.random.choice(v_spec.shape[0], N)
    v_spec_random = v_spec[choice]
    data = np.vstack((v_spec_random, a_spec))
    # create dataset and labels(v: 0, a:1)
    label = label = np.array([0] * N + [1] * N)
    return data, label



def cross_validation(data, label, n=10):
    """
    return list of train_data, test_data, train_label, test_label
    return
        list: list of shape(n, 4)
    """
    index = np.arange(data.shape[0])
    index = index % n
    cvindex = make_cvindex(index)
    cvindex = np.array(cvindex)
    cross_valid = []
    for i in range(cvindex.shape[2]):
        train_data = data[cvindex[0][:, i] == True]
        test_data = data[cvindex[1][:, i] == True]
        train_label = label[cvindex[0][:, i] == True]
        test_label = label[cvindex[1][:, i] == True]
        cross_valid.append((train_data, test_data, train_label, test_label))
    return cross_valid



# use linar SVM and calculate the score
def classify(train_data, test_data, train_label, test_label, n_features):
    # scale data
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    # train classifier
    clf = LinearSVC()
    # use n_features
    clf.fit(scaled_train_data[:, : n_features], train_label)
    # get score
    score = clf.score(scaled_test_data[:, :n_features], test_label)
    return score

if __name__  == "__main__":
    # load data"
    v_spec = np.load("ventricle_spectrum.npy")
    a_spec = np.load("article_spectrum.npy")

    # put data to score_record and get sum 
    print("A")
    score_record = np.zeros(len(NUM_FEATURES))
    for i in range(10):
        print("iteration number{0}".format(i))
        data, label = make_dataset(v_spec, a_spec)
        cross_valid = cross_validation(data, label)
        # for each train-test in cross validation
        for  train_data, test_data, train_label, test_label in cross_valid:
            for count, num_features in enumerate(NUM_FEATURES):
                score = classify(train_data, test_data, train_label, test_label, num_features)
                score_record[count] += score

