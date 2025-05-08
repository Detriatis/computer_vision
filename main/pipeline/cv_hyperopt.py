from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from main.utils.scikit_interface import ImageNormalizer, DescriptorExtractor, BoWTransformer, SKlearnPyTorchClassifier
from main.pipeline.cnn import Net
from main.utils.image_loading import load_directory_images

import logging
from joblib import Memory 

from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

import cv2
import pandas as pd  
import numpy as np 
from pathlib import Path
import multiprocessing as mp

mp.set_start_method('forkserver', force=True)

# Norm Paramaters
search_space_norms = [
            cv2.NORM_MINMAX,
            'eq_hist',
            ]

kp_desc = [
    ('none', 'none'),
    ('sift', 'sift'),
    ('sift', 'dense_hist'),
    ('dense', 'sift'),
    ('dense', 'dense_hist'),
]

detector_params = {
    'sift' : {},
    'dense': {
        'kp_size': 1,
        'step_size': 5,
    },
    'none': {},
}
# Bag of Words Parameters
search_space_bow = {
    'k': 0,
    'max_iters':1000,
    'batch':1024,
}
words = [50, 100, 200]


# Classifier Parameters
search_space_rf = {
        'n_estimators': Integer(50 ,500),
        'max_depth': Integer(5, 50), 
        'min_samples_split': Integer(2, 20),
}

search_space_knn = {
        'n_neighbors': Integer(10, 200),
        'weights': Categorical(['uniform', 'distance']),
        'p': Integer(1, 2),
}


search_space = {
 
    'rf': {
        'classifier' : RandomForestClassifier(),
        'parameter_space' : search_space_rf
            },   
  
    'knn': {
        'classifier' : KNeighborsClassifier(), 
        'parameter_space' : search_space_knn
            },
 
  }

directories = ['mammals', 'stl10'] 
for dir in directories:
    print(dir)
    DIRECTORYPATH = Path("/home/michaeldodds/Projects/manchester/computer_vision/datasets/processed/") / dir
    WRITEOUTPATH = "/home/michaeldodds/Projects/manchester/computer_vision/results"

    imgs = load_directory_images(DIRECTORYPATH, train = True, subsample=100) 
    X = imgs.imgs
    print(f'Training on {len(X)}')
    
    for norm_type in search_space_norms:

        imagenormalizer = ImageNormalizer(norm_type=norm_type).fit(X)
        norm_x = imagenormalizer.transform(X)
        
        for kp, desc in kp_desc: 
            descriptorextractor = DescriptorExtractor(desc_algo=desc, kp_algo=kp, **detector_params[kp]).fit(norm_x)
            transform_x = descriptorextractor.transform(norm_x)
            
            for k in words:
                if kp != 'none':
                    search_space_bow['k'] = k 
                    bowtransformer = BoWTransformer(**search_space_bow).fit(np.concat(transform_x))
                    bows = bowtransformer.transform(transform_x)
                else:
                    k = 'None' 
                    bows = np.array(transform_x).reshape(-1, 96 * 96)

                for key, value in search_space.items():
                    
                    print(norm_type, desc, kp, k, key, dir)
                    classifier = value['classifier']
                    parameter_space = value['parameter_space']
                    
                    opt = BayesSearchCV(
                        classifier,
                        parameter_space,
                        n_iter=16,
                        scoring='accuracy',
                        cv=3,
                        verbose=1,
                        random_state=1,
                        n_jobs=-1,
                        n_points=1
                    )
                    opt.fit(bows, imgs.labels)

                    result_df = pd.DataFrame(opt.cv_results_)[[
                        'params', 'mean_test_score', 'std_test_score'
                        ]].sort_values('mean_test_score', ascending=False)        
                    result_df['norm_type'] = norm_type
                    result_df['descriptor_algo'] = desc 
                    result_df['keypoints_algo'] = kp 
                    result_df['k'] = k
                    result_df['classifier'] = key
                    result_df['dataset'] = dir
                    
                    iterative_path = Path(WRITEOUTPATH, 'hyperopt', 'final_iterative_exps.csv')
                    if iterative_path.exists():
                        df = pd.read_csv(iterative_path)
                        iterative_df = pd.concat([result_df, df])
                        iterative_df.to_csv(iterative_path, index=False)
                    else:     
                        result_df.to_csv(iterative_path, index=False)
                print(kp) 
                if kp == 'none':
                    print('breaking loop')
                    break