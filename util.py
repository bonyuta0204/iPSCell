
# coding: utf-8
"""
module for providing some useful functions
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/home/share/libraries/bdpy")
import bdpy
from bdpy.preproc import select_top
from bdpy.ml import add_bias, make_cvindex
from bdpy.stats import corrcoef


# choose 71 samples from vertricle and make test data and label data
def make_dataset(v_spec, a_spec, shuffle=False):
    """
    randomely choose 71 samples from vertricle and make test data and label data
    parameter:
        v_spec: np.array. ventricle data(n * features)
        a_spec: np.array. article data(n * features)
        shuffle: Bool. default = False
            if True, shuffle the data
    return:
        (data, label)
        data: np.array 
        label: np.array
            label for dataset. 0 for ventricle and 1 for article
    """
    N = np.min((a_spec.shape[0], v_spec.shape[0]))
    v_random = np.random.permutation(v_spec)
    v_random = v_random[:N]

    a_random = np.random.permutation(a_spec)
    a_random = a_random[:N]

    data = np.vstack((v_random, a_random))

    # create dataset and labels(v: 0, a:1)
    label = label = np.array([0] * N + [1] * N)
    # shuffle data and label if shuffle == True
    if shuffle:
        random_index = [i for i in range(data.shape[0])]
        random.shuffle(random_index)
        data = data[random_index]
        label = label[random_index]
    return data, label


def cross_validation(data, label, n=10):
    """
    generator
    yield train_data, test_data, train_label, test_label for given data, label

    parameter:
        data: np.array
        label: np.array
        n: int
            number of fold
    return:generator
           this generator returns train_data, test_data, train_label, test_label 
           for n times
    """
    index = np.arange(data.shape[0])
    index = index % n
    cvindex = make_cvindex(index)
    cvindex = np.array(cvindex)
    for i in range(cvindex.shape[2]):
        train_data = data[cvindex[0][:, i] == True]
        test_data = data[cvindex[1][:, i] == True]
        train_label = label[cvindex[0][:, i] == True]
        test_label = label[cvindex[1][:, i] == True]
        yield train_data, test_data, train_label, test_label


# for testing this module
if __name__ == "__main__":
    v_spec = np.load("record/ventricle_spectrum.npy")
    a_spec = np.load("record/article_spectrum.npy")
    data, label = make_dataset(v_spec, a_spec)
    for train_data, test_data, train_label, test_label in cross_validation(data, label):
        print("a")
