import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from main.utils.cv_dataclasses import ProcessedImages, Images
from main.utils.image_loading import load_directory_images, load_labeled_images

def normalization(normaliser, parameters, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return normaliser(img, None, **parameters)

def compute_descriptors(detector, img): 
    features, descriptors = detector.detectAndCompute(img, mask=None)
    return features, descriptors

def calc_features_desc(imgs: Images, normaliser, normaliser_parameters, detector = None, descriptor = None) -> ProcessedImages:
 
    all_feats, all_descs, all_norm_imgs = [], [], []  
 
    for img in imgs.imgs:
        
        norm_img = normalization(normaliser, 
                                 normaliser_parameters, 
                                 img)
     
        kps = detector.detect(img, mask=None)
        kps, descs = descriptor.compute(img, kps) 
        
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