import sys
import csv
import numpy as np
import sklearn


#SVM functions
from sklearn import svm
from sklearn import ensemble
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)

    set = np.loadtxt('../data/train.csv', delimiter=',')
    train_set = set[:,1:7]
    train_labels = set[:,8]

    validation = np.loadtxt('../data/validate_and_test.csv', delimiter=',')
    ids = validation[:,0]
    valid = validation[:,1:7]

    # Preprocessing

    # SVM with RBF Kernel

    # train_set = SelectKBest(chi2, k=2).fit_transform(X, y)

    clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=2, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    clf.fit(train_set, train_labels)

    #Prediction

    svm_predicions = clf.predict(valid)

    np.savetxt("predict_svm.csv", np.c_[ids, svm_predicions], delimiter=",", fmt='%d')


    # Random Forest

    rf = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=None,
                                         min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                         max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                         oob_score=False, n_jobs=1, random_state=None, verbose=0,
                                         warm_start=False, class_weight=None)

    rf.fit(train_set, train_labels)
    rf_predicions = rf.predict(valid)

    np.savetxt("predict_rf.csv", np.c_[ids, rf_predicions], delimiter=",", fmt='%d')













