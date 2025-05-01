def fit_model(clf, train_x, train_y):
    clf.fit(train_x, train_y)

def pred_model(clf, test_x):
    preds = clf.predict(test_x)

def model_accuracy(loss_metric, preds, test_y):
    # Generally the loss metric 
    # used to fit should be used to evaluate
    return loss_metric(test_y, preds)  