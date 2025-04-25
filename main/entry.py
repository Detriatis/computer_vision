import sys
import numpy as np
import cv2

from pathlib import Path
from main.utils.image_loading import load_labeled_images, load_directory_images
from main.pipeline.feature_detection import calc_features_desc
from main.pipeline.clustering import bag_of_words 
from main.pipeline.classifier import fit_model
    
DIRECTORYPATH = Path("/home/michaeldodds/Projects/manchester/computer_vision/datasets/processed/stl10/train/")
PARAMETERS = {  'normaliser': {'alpha': 0, 'beta': 255, 'norm_type': cv2.NORM_MINMAX},
                'detector': {'nfeatures': 1000},
                'bag_of_words': {'k': 50, 'max_iters': 1000},
                } 

if __name__ == '__main__':
    normaliser = cv2.normalize
    detector = cv2.SIFT_create(**PARAMETERS['detector'])
    imgs = load_directory_images(DIRECTORYPATH) 
    t_imgs = load_directory_images(DIRECTORYPATH.with_name('test'))
    p_imgs = calc_features_desc(imgs, PARAMETERS, normaliser, detector, None)
    tp_imgs = calc_features_desc(t_imgs, PARAMETERS, normaliser, detector, None)
    
    bow, means = bag_of_words(p_imgs, **PARAMETERS['bag_of_words'])
    t_bow, _ = bag_of_words(tp_imgs, **PARAMETERS['bag_of_words'], means=means)

    preds = fit_model(None, bow, imgs.labels, t_bow)
    print(sum(preds == t_imgs.labels))
    print(len(preds))