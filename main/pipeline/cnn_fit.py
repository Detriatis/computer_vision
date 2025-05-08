import numpy as np
import pandas as pd 
from collections import OrderedDict
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from main.pipeline.cnn_hyperopt import Net, NetSmaller
from main.utils.scikit_interface import ImageNormalizer, DescriptorExtractor, BoWTransformer, get_sklearn_metric, SKlearnPyTorchClassifier
from main.utils.image_loading import load_directory_images
from main.utils.cv_dataclasses import Images

import torchvision.transforms as transforms 
import cv2


def pipeline(clf, imgs: Images, row = None, bowtransformer = None, train: bool = True):
    X = imgs.imgs
    if row is not None: 
        try: 
            norm = int(row['norm_type'])
        except ValueError: 
            norm = str(row['norm_type'])
      
        if not isinstance(clf, SKlearnPyTorchClassifier):
            desc_algo = row['descriptor_algo']
            kp_algo = row['keypoints_algo']
       
            if kp_algo != 'none':
                k = int(row['k'])
       
        else:
            col = row['colour']
            if str(col) == '6.0':
                col = cv2.COLOR_BGR2GRAY
            else: 
                col = None 
            bowtransformer = None
            descriptorextractor = None 
            desc_algo = None
            kp_algo = None
        
     
    imagenormalizer = ImageNormalizer(norm_type=norm, alpha=0, beta=255, dtype=np.uint8, col=col).fit(imgs.imgs)
    X = imagenormalizer.transform(X)

    if not isinstance(clf, SKlearnPyTorchClassifier):  
        descriptorextractor = DescriptorExtractor(kp_algo=kp_algo, desc_algo=desc_algo).fit(X)
        X = descriptorextractor.transform(X)
        
        if kp_algo != 'none':
            if train:
                bowtransformer = BoWTransformer(k=k).fit(np.concat(X))
            
            bows = bowtransformer.transform(X)
        
        else:
            bows = np.array(X).reshape(-1, 96 * 96) 
    
    elif col == cv2.COLOR_BGR2GRAY:  
        bows = np.expand_dims(X, -1)
    else: 
        bows = np.array(X)
    
    if train:
        clf.fit(bows, np.array(imgs.labels))
    
    preds = clf.predict(bows)

    return clf, preds, bowtransformer, descriptorextractor 

def class_label_assigments(labels, kps, accuracy_scores):
    label_dic = {str(label): {'n_keypoints': 0, 'accuracy': 0} for label in sorted(np.unique(labels))}
    if kps is not None: 
        for kp, label in zip(kps, labels):
            label_dic[str(label)]['n_keypoints'] += len(kp) 
    
    for i, score in enumerate(accuracy_scores):
        label_dic[str(i)]['accuracy'] = score 
    return label_dic

if __name__ == '__main__':
    
    DIRECTORYPATH  = Path("/home/michaeldodds/Projects/manchester/computer_vision/datasets/processed/")
    INPUTPATH = Path("/home/michaeldodds/Projects/manchester/computer_vision/results/hyperopt/aggregated_results/")
    RESULTPATH = Path("/home/michaeldodds/Projects/manchester/computer_vision/results/")
    
    norm_tuple = (0.5, 0.5, 0.5)
    image_resize = (96, 96) 
    
    transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(norm_tuple, norm_tuple),
                transforms.Resize(image_resize)]
            )
    classifiers = {
            'rf' : RandomForestClassifier(),
            'knn' : KNeighborsClassifier(),
            'cnn' : SKlearnPyTorchClassifier(NetSmaller, device='cuda', transform=transform, image_size = (96, 96)),
    }
    
    accuracy = get_sklearn_metric('accuracy')
    
    test_bool = False 
    
    feature_params = {} 
    norm_params = {} 
    bow_params = {}
    classifier_params = {}
    rows = [] 
    classifiers_df = pd.read_csv(INPUTPATH / 'cnn_best_classifiers.csv')
    
    for i in range(len(classifiers_df)): 
        row = classifiers_df.iloc[i].copy() 
        print(row)
        directory = row['dataset'] 
        
        imgs = load_directory_images(DIRECTORYPATH / directory, train=True)
        test_imgs = load_directory_images(DIRECTORYPATH / directory, train=False)

        accuracy = get_sklearn_metric('accuracy')
        
        classifier = row['classifier']  
        if classifier == 'cnn':
            imgs.labels = np.array(imgs.labels) - 1
            test_imgs.labels = np.array(test_imgs.labels) - 1

        classifier_params =  eval(row['params'], {'OrderedDict' : OrderedDict})
        
        clf = classifiers[classifier]
        clf = clf.set_params(**classifier_params)        
        
        if not test_bool: 
            clf, train_preds, bowtransformer, _ = pipeline(clf, imgs, row, train=True)    
            clf, test_preds, _, descriptorextractor =  pipeline(clf, test_imgs, row, bowtransformer=bowtransformer, train=False)

            if descriptorextractor:
                kps_list = descriptorextractor.kps
                descs_list = descriptorextractor.descs
            
            train_score = accuracy(imgs.labels, train_preds) 
            test_score = accuracy(test_imgs.labels, test_preds) 
            
            cm = confusion_matrix(test_imgs.labels, test_preds)
            if descriptorextractor:
                results = class_label_assigments(test_imgs.labels, kps_list, (cm.diagonal() / cm.sum(axis=1)))
            else: 
                results = class_label_assigments(test_imgs.labels, None, (cm.diagonal() / cm.sum(axis=1)))
            
            results_df = pd.DataFrame(results)

            print(f'Score on training set {train_score} with classifier {classifier}')
            print(f'Score on testing set {test_score} with classifier {classifier}')
            row['fullmodelaccuracy'] = accuracy
            row['index'] = i
            rows.append(row)


            outpath = Path('/home/michaeldodds/Projects/manchester/computer_vision/results/fullmodel', f'{classifier}_{str(i)}')
            outpath.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(outpath / 'cnn_class_accuracy.csv', index=False)
            np.savetxt(outpath / "confusion_matrix.csv", cm, delimiter=",") 
    
    rowsframe = pd.concat(rows, axis=1).T
    rowsframe.to_csv(outpath.parent / 'cnn_fullmodel.csv', index=False)

