import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from main.utils.cv_dataclasses import ProcessedImages, Images
from main.utils.image_loading import load_directory_images, load_labeled_images

def normalization(normaliser, parameters, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return normaliser(img, None, **parameters)

def dense_keypoints(img, step_size):
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
                                        for x in range(0, img.shape[1], step_size)]
    return kp 

def compute_descriptors(detector, img): 
    features, descriptors = detector.detectAndCompute(img, mask=None)
    return features, descriptors

def calc_features_desc(imgs: Images, parameters: dict, normaliser, detector = None, descriptor = None) -> ProcessedImages:
 
    all_feats, all_descs, all_norm_imgs = [], [], []  
 
    for img in imgs.imgs:
        
        norm_img = normalization(normaliser, 
                                 parameters['normaliser'], 
                                 img)
     
        if not descriptor and detector: 
            kps, descs = detector.detectAndCompute(img, mask=None)
        if descriptor and not detector:
            # We use dense keypoints as an alternative to feature detection 
            kp = dense_keypoints(img, **parameters['keypoints'])
            kp, des = descriptor.compute(img, kp)
        if descriptor and detector:
            kp = detector.detect(img, mask=None)
            kp, des = descriptor.compute(img, kp) 
        
        all_norm_imgs.append(norm_img)
        all_feats.append(kps)
        all_descs.append(descs)

    return ProcessedImages(
        norm_imgs =     all_norm_imgs,
        keypoints =      all_feats,
        descriptors =   all_descs,
        descriptor_type= type(descriptor),
        train = imgs.train
        ) 