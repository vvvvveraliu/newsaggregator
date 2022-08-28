import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from pre_process import split_dataset
import joblib


if __name__ == '__main__':
    x = np.load('processed/bow_x.npy')
    y = np.load('processed/bow_y.npy')
    dataset = split_dataset((x, y))
    train_set, val_set, test_set = dataset
    print(train_set[0].shape, train_set[1].shape)
    print(val_set[0].shape, val_set[1].shape)
    print(test_set[0].shape, test_set[1].shape)

    clf = joblib.load('random_forest.model')
    print(classification_report(test_set[1], clf.predict(test_set[0])))
    print(confusion_matrix(test_set[1], clf.predict(test_set[0])))
