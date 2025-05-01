from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import log_loss
from skopt import BayesSearchCV

searchspace = { 
    'supportvectormachines': {
        'c' : (1, 2)
                              }, 
    'randomforest': {
        'n_estimators': (50 ,500),
        'max_depth': (5, 50), 
        'min_samples_split': (2, 20) 
                     },

    'knn': {
        'n_neighbours': (2, 30),
    } 
}

models_to_compare = {
    'supportvectormachines': LinearSVC,
    'randomforest': RandomForestClassifier,
    'knn': NearestNeighbors 
}

def fit_model(clf, train_x, train_y):
    clf.fit(train_x, train_y)

def pred_model(clf, test_x):
    preds = clf.predict(test_x)

def model_accuracy(loss_metric, preds, test_y):
    # Generally the loss metric 
    # used to fit should be used to evaluate
    return loss_metric(test_y, preds)  