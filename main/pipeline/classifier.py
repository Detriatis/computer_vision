from sklearn.svm import LinearSVC



def fit_model(clf, bows_train, labels_train, bows_test):
    clf = LinearSVC(C=1.0)   
    clf.fit(bows_train, labels_train)
    preds = clf.predict(bows_test)
    return preds