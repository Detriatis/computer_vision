import numpy as np 
import pandas as pd 
from collections import OrderedDict
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from main.utils.scikit_interface import ImageNormalizer, DescriptorExtractor, BoWTransformer, get_sklearn_metric
from main.utils.image_loading import load_directory_images
from main.utils.cv_dataclasses import Images


def class_label_assigments(labels, kps, accuracy_scores):
    label_dic = {str(label): {'n_keypoints': 0, 'accuracy': 0} for label in sorted(np.unique(labels))}
    if kps is not None: 
        for kp, label in zip(kps, labels):
            label_dic[str(label)]['n_keypoints'] += len(kp) 
    
    for i, score in enumerate(accuracy_scores):
        label_dic[str(i + 1)]['accuracy'] = score 
    return label_dic




def pipeline(clf, imgs: Images, row = None, bowtransformer = None, train: bool = True):
    X = imgs.imgs
    if row is not None: 
        try: 
            norm = int(row['norm_type'])
        except ValueError: 
            norm = str(row['norm_type'])
        desc_algo = row['descriptor_algo']
        kp_algo = row['keypoints_algo']
        if kp_algo != 'none':
            k = int(row['k'])
    
    imagenormalizer = ImageNormalizer(norm).fit(X)
    X = imagenormalizer.transform(X)
    
    descriptorextractor = DescriptorExtractor(kp_algo=kp_algo, desc_algo=desc_algo).fit(X)
    X = descriptorextractor.transform(X)
    if kp_algo != 'none':
        if train:
            bowtransformer = BoWTransformer(k=k).fit(np.concat(X))
        
        bows = bowtransformer.transform(X)
    else:
        bows = np.array(X).reshape(-1, 96 * 96) 

    if train:
        clf.fit(bows, imgs.labels)
    
    preds = clf.predict(bows)

    return clf, preds, bowtransformer, descriptorextractor 

DATA_PARAMETERS  = {
    'directorypath': Path("/home/michaeldodds/Projects/manchester/computer_vision/datasets/processed/"),
    'bestclassifierpath': "/home/michaeldodds/Projects/manchester/computer_vision/results/hyperopt/aggregated_results/cv_best_classifiers.csv",
    'writeoutpath': "/home/michaeldodds/Projects/manchester/computer_vision/results/"
                    }

classifiers = {
        'rf' : RandomForestClassifier(),
        'knn' : KNeighborsClassifier()
}

if __name__ == '__main__':
    test_bool = False 

    classifiers_df = pd.read_csv(DATA_PARAMETERS['bestclassifierpath'])

    feature_params = {} 
    norm_params = {} 
    bow_params = {}
    classifier_params = {}
    rows = [] 
    
    for i in range(len(classifiers_df)):    
        row = classifiers_df.iloc[i].copy() 
        
        imgs = load_directory_images(DATA_PARAMETERS['directorypath'] / row.dataset, train=True)
        test_imgs = load_directory_images(DATA_PARAMETERS['directorypath'] / row.dataset, train=False)

        accuracy = get_sklearn_metric('accuracy')
        
        classifier = row['classifier']  
        classifier_params =  eval(row['params'], {'OrderedDict' : OrderedDict})
        
        clf = classifiers[classifier]
        clf = clf.set_params(**classifier_params)        
        
        if not test_bool: 
            clf, train_preds, bowtransformer, _ = pipeline(clf, imgs, row, train=True)    
            clf, test_preds, _, descriptorextractor =  pipeline(clf, test_imgs, row, bowtransformer=bowtransformer, train=False)

            kps_list = descriptorextractor.kps
            descs_list = descriptorextractor.descs
            
            train_score = accuracy(imgs.labels, train_preds) 
            test_score = accuracy(test_imgs.labels, test_preds) 
            
            cm = confusion_matrix(test_imgs.labels, test_preds)
            results = class_label_assigments(test_imgs.labels, kps_list, (cm.diagonal() / cm.sum(axis=1)))
            
            results_df = pd.DataFrame(results)

            print(f'Score on training set {train_score} with classifier {classifier}')
            print(f'Score on testing set {test_score} with classifier {classifier}')
            row['fullmodelaccuracy'] = test_score 
            row['index'] = i
            rows.append(row)


            outpath = Path('/home/michaeldodds/Projects/manchester/computer_vision/results/fullmodel', f'{classifier}_{str(i)}')
            outpath.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(outpath / 'class_accuracy.csv', index=False)
            np.savetxt(outpath / "confusion_matrix.csv", cm, delimiter=",")    
    
    rowsframe = pd.concat(rows, axis=1).T
    rowsframe.to_csv(outpath.parent / 'cv_fullmodel.csv', index=False)

