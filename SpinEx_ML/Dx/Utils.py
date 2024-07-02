import csv
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score


def get_labels_n_cancerProbs(label, cancer_probs):
	return 0



# Define class label from regression result.
def get_label_from_score(pred):
    temp = []
    for reg_value in pred:
        if reg_value > 0.5:
            reg_value = 1
            temp.append(reg_value)
        else:
            reg_value = 0
            temp.append(reg_value)
    return np.array(temp)


# Define class label from class name.
def get_label_from_class(pred):
    temp = []
    for reg_value in pred:
        if reg_value > 0.5:
            reg_value = 1
            temp.append(reg_value)
        else:
            reg_value = 0
            temp.append(reg_value)
    return np.array(temp)


# Save data to a new CSV file.
def save_data_2csv(filename, data):
    with open(filename, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(data)


# Append new record to existed CSV file.
def record_data_2csv(filename, data):
    with open(filename, 'a', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(data)


# Read result data from csv.
def read_csv(filename):
    with open(filename, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_feature_index(marker_index, exp_index, feature_type):
    feature_index = marker_index*6 + exp_index*3 + feature_type
    return feature_index


def get_feature_index_4ratio(marker_index, feature_type):
    feature_index = marker_index*3 + feature_type
    return feature_index


def check_nan(values):
    if np.isnan(values):
        return 0
    else:
        return values


def get_list_column(list2d, col_index):
    col_list = []
    for row_index in range(len(list2d)):
        col_list.append(list2d[row_index][col_index])
    return col_list


def get_ftname_from_index(index):
    marker_index = index // 6
    exp_index = (index % 6) // 3
    if exp_index == 0:
        exp_name = 'F'
    else:
        exp_name = 'P'
    feature_type_index = (index % 6) % 3
    if feature_type_index == 0:
        feature_type = 'mean'
    elif feature_type_index == 1:
        feature_type = 'std'
    else:
        feature_type = 'median'

    return marker_index, exp_name, feature_type


def get_ftratio_from_index(index, multiplier):
    marker_index = index // multiplier
    feature_type_index = index % multiplier
    if feature_type_index == 0:
        feature_type = 'mean'
    elif feature_type_index == 1:
        feature_type = 'std'
    else:
        feature_type = 'median'

    return marker_index, feature_type




def get_clf_result(classifier, test_data, test_label):
    clf_predict = classifier.predict_proba(test_data)
    predcs1 = clf_predict[:, 1]

    fpr, tpr, threshold = metrics.roc_curve(test_label, predcs1)
    roc_auc = metrics.auc(fpr, tpr)

    prediction = np.argmax(clf_predict, axis=1)

    cm = metrics.confusion_matrix(test_label, prediction)
    correct = (prediction == test_label).mean()
    f1score = metrics.f1_score(test_label, prediction)
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    return cm, roc_auc, f1score, sensitivity, specificity, correct, fpr, tpr, threshold, clf_predict


# Br is 0, Cl is 1, Li is 2, Lu is 3, Pa is 4
def convert_multilabel(label):
    if label == 0:
        str_label = 'Br'
    elif label == 1:
        str_label = 'Cl'
    elif label == 2:
        str_label = 'Li'
    elif label == 3:
        str_label = 'Lu'
    elif label == 4:
        str_label = 'Pa'
    else:
        str_label = 'unverified'

    return str_label

