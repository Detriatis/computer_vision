import numpy as np 
import pandas as pd 
from collections import OrderedDict

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from main.utils.scikit_interface import ImageNormalizer, DescriptorExtractor, BoWTransformer, get_sklearn_metric
from main.utils.image_loading import load_directory_images

DATA_PARAMETERS  = {
    'directorypath': "/home/michaeldodds/Projects/manchester/computer_vision/datasets/processed/stl10",
    'bestclassifierpath': "/home/michaeldodds/Projects/manchester/computer_vision/results/hyperopt/best_classifiers.csv",
    'writeoutpath': "/home/michaeldodds/Projects/manchester/computer_vision/results/"
                    }


norm_params =  {'norm_type': None}
feature_extractor_params = {'desc_algo' : None, 'kp_algo': None, 'detector_params': {}}
bow_params = {'k': None} 
classifier = 'rf' 

classifiers = {
        'rf' : RandomForestClassifier(),
        'svc' : SVC(),
        'knn' : KNeighborsClassifier()
}

classifier_params = {
        'rf': {},
        'svc': {},
        'knn': {},
}

clf = classifiers[classifier]

def pipeline(clf, X, bowtransformer = None, train: bool = True):
        
    imagenormalizer = ImageNormalizer(**norm_params).fit(X)
    X = imagenormalizer.transform(X)
    
    descriptorextractor = DescriptorExtractor(desc_algo=feature_extractor_params['desc_algo'], kp_algo=feature_extractor_params['kp_algo'], **feature_extractor_params['detector_params']).fit(X)
    X = descriptorextractor.transform(X)
    
    if train:
        bowtransformer = BoWTransformer(**bow_params).fit(np.concat(X))
    
    bows = bowtransformer.transform(X)

    if train: 
        clf.fit(bows, imgs.labels)
    
    preds = clf.predict(bows)

    return clf, preds, bowtransformer

if __name__ == '__main__':
    test_bool = False
    classifiers_df = pd.read_csv(DATA_PARAMETERS['bestclassifierpath'])
    
    imgs = load_directory_images(DATA_PARAMETERS['directorypath'], train=True)
    test_imgs = load_directory_images(DATA_PARAMETERS['directorypath'], train=False)
    for classifier in classifiers.keys(): 
        params_df = classifiers_df[classifiers_df['classifier'] == classifier].iloc[0]
        classifier_params[classifier] =  eval(params_df['params'], {'OrderedDict' : OrderedDict})
        
        clf = classifiers[classifier]
        clf.set_params(**classifier_params[classifier])
        
        feature_extractor_params['desc_algo'] = params_df['descriptor_algo']
        feature_extractor_params['kp_algo'] = params_df['keypoints_algo']
        feature_extractor_params['detector_params']

        norm_params['norm_type'] = params_df['norm_type']
        bow_params['k'] = params_df['k']

        accuracy = get_sklearn_metric('accuracy')

        if not test_bool: 
                        
            print(f'Testing Pipeline with classifier {classifier}')
            clf, train_preds, bowtransformer = pipeline(clf, imgs.imgs, True)    
            clf, test_preds, _ = pipeline(clf, test_imgs.imgs, bowtransformer, False)
            train_score = accuracy(imgs.labels, train_preds) 
            test_score = accuracy(test_imgs.labels, test_preds) 
            print(f'Score on training set {train_score} with classifier {classifier}')
            
            print(f'Score on testing set {test_score} with classifier {classifier}')