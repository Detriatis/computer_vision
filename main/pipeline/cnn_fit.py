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


def pipeline(clf, imgs: Images, row = None, bowtransformer = None, train: bool = True, test_imgs: Images = None):
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
            bowtransformer = None
            descriptorextractor = None 
            desc_algo = None
            kp_algo = None
         

            colour = row['colour']
            
            if str(colour) != 'rgb':
                col = cv2.COLOR_BGR2GRAY
                norm_tuple = (0.5)
            else: 
                col = None 
                norm_tuple = (0.5, 0.5, 0.5)
    
    
    image_resize = (96, 96) 
    transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(norm_tuple, norm_tuple),
                transforms.Resize(image_resize)]
            )
    
    if isinstance(clf, SKlearnPyTorchClassifier):
        clf.transform = transform
     
    imagenormalizer = ImageNormalizer(norm_type=norm, alpha=0, beta=255, dtype=np.uint8, col=col).fit(imgs.imgs)
    X = imagenormalizer.transform(X)
    if test_imgs:
        x_test = imagenormalizer.transform(test_imgs.imgs)

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

    if test_imgs:
        if col == cv2.COLOR_BGR2GRAY:
            x_test = np.expand_dims(np.array(x_test), -1)
        
        clf.x_test = x_test
        clf.y_test  = np.array(test_imgs.labels)
    
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
    
    DIRECTORYPATH  = Path("/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/")
    INPUTPATH = Path("/home/REDACTED/Projects/manchester/computer_vision/results/hyperopt/aggregated_results/")
    RESULTPATH = Path("/home/REDACTED/Projects/manchester/computer_vision/results/")
    
    classifiers = {
            'rf' : RandomForestClassifier(),
            'knn' : KNeighborsClassifier(),
            'split_cnn' : SKlearnPyTorchClassifier(NetSmaller, device='cuda', image_size = (96, 96)),
            'classic_cnn' : SKlearnPyTorchClassifier(Net, device='cuda', image_size = (96, 96)),
    }
    
    accuracy = get_sklearn_metric('accuracy')
    
    test_bool = False 
    quick_test = False 
    
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
        
        imgs = load_directory_images(DIRECTORYPATH / directory, train=True, subsample=0)
        test_imgs = load_directory_images(DIRECTORYPATH / directory, train=False, subsample=0)

        accuracy = get_sklearn_metric('accuracy')

        # Classifier SETUP 
        classifier = row['classifier']  
        if str(classifier).endswith('cnn'):
            imgs.labels = np.array(imgs.labels) - 1
            test_imgs.labels = np.array(test_imgs.labels) - 1
        classifier_params =  eval(row['params'], {'OrderedDict' : OrderedDict})
        clf = classifiers[classifier] 
        if quick_test:
            classifier_params['epochs'] = 3
        clf = clf.set_params(**classifier_params)        
        
        if not test_bool:
            # TRAINING AND FITTING ---- 
            clf, train_preds, bowtransformer, _ = pipeline(clf, imgs, row, train=True, test_imgs=test_imgs)    
            clf, test_preds, _, descriptorextractor =  pipeline(clf, test_imgs, row, bowtransformer=bowtransformer, train=False)
            
            if isinstance(clf, SKlearnPyTorchClassifier):
                history = clf.history

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
            
            row['fullmodelaccuracy'] = test_score
            row['index'] = i
            rows.append(row)


            outpath = Path('/home/REDACTED/Projects/manchester/computer_vision/results/fullmodel', f'{classifier}_{str(i)}')
            outpath.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(outpath / 'cnn_class_accuracy.csv', index=False)
            
            pd.DataFrame(history).to_csv(outpath / 'cnn_training_history.csv', index=False)
            np.savetxt(outpath / "confusion_matrix.csv", cm, delimiter=",") 

        if quick_test:
            break 

    rowsframe = pd.concat(rows, axis=1).T
    rowsframe.to_csv(outpath.parent / 'cnn_fullmodel.csv', index=False)

