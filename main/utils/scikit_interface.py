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

_CLASSIFIERS = {
    'svm': LinearSVC,
    'rf': RandomForestClassifier,
    'knn': NearestNeighbors 
}

_METRICS = {
    'ce': log_loss,
    ''
}

def get_sklearn_detector(name: str, **kwargs):
    func = _CLASSIFIERS.get(name.lower())
    if func is None: 
        raise ValueError(f"No OpenCV detector named {name!r}.\n"
                         f"Available Detectors:\n----\n{"\n".join(list(_CLASSIFIERS))}")
    
    return func(**kwargs)
