import sys
import csv
import numpy as np
import sklearn


#SVM functions
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

if __name__ == '__main__':

    set = np.loadtxt('../data/train.csv', delimiter=',')
    train_set = set[:,1:7]
    train_labels = set[:,8]

    X, y = train_set, train_labels
    print train_set.size


    # Preprocessing

    # train_set = SelectKBest(chi2, k=2).fit_transform(X, y)


    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=2, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    clf.fit(train_set, train_labels)

    #Prediction

    validation = np.loadtxt('../data/validate_and_test.csv', delimiter=',')

    valid = validation[:,1:7]

    predicions = clf.predict(valid)

    print predicions
