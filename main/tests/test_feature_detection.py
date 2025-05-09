from main.pipeline.feature_detection import calc_features_desc
from main.utils.image_loading import load_labeled_images
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    directorypath = Path("/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/cifar-10/train/1")
    parameters = {'normaliser': {'alpha': 0, 'beta': 255, 'norm_type': cv2.NORM_MINMAX},
                  'detector': {'nfeatures': 1000}} 
    
    normaliser = cv2.normalize
    detector = cv2.SIFT_create(**parameters['detector'])
    descriptor = None

    imgs = load_labeled_images(directorypath)

    imgs, procced_imgs = calc_features_desc(imgs, 
                                      parameters, 
                                      normaliser, 
                                      detector, 
                                      descriptor
                                      )

      
    assert len(procced_imgs.norm_imgs) == len(imgs.imgs)

    i = 20

    img = imgs.imgs[i]
    label = imgs.labels[i]
    norm_img = procced_imgs.norm_imgs[i]
    kps = procced_imgs.features[i]
    img_kp = cv2.drawKeypoints(norm_img, kps, None, color=(0, 255, 0), flags=0)
    plt.imshow(img_kp), plt.show() 