from __future__ import print_function

import sys
import pickle 
from pathlib import Path 
import os, sys, tarfile, errno
import numpy as np
import matplotlib.pyplot as plt
    
if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib # ugly but works
else:
    import urllib

try:
    from imageio import imsave
except:
    from scipy.misc import imsave

print(sys.version_info) 

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = Path('/home/REDACTED/Projects/manchester/computer_vision/datasets/raw/stl10/data/stl10_binary')

# path to the binary train file with image data
TRAIN_DATA_PATH = DATA_DIR / 'train_X.bin'

# path to the binary train file with labels
TRAIN_LABEL_PATH = DATA_DIR / 'train_y.bin'

# path to binary test file with image data
TEST_DATA_PATH = DATA_DIR / 'test_X.bin'

# path to the binary train file with labels
TEST_LABEL_PATH = DATA_DIR / 'test_y.bin'

# path to the output file for images 
OUTPUT_DIR = Path("/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/stl10")

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()

def save_image(image, name):
    imsave("%s.png" % name, image, format="png")


def save_images(images, labels, filepath: Path):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        
        directory = filepath / str(label)
        
        if not directory.exists():
            directory.mkdir(parents=True)
        
        filename = directory / str(i)
        
        print(filename)
        save_image(image, filename)
        i = i+1

def read_class_names(filepath: Path): 
    with open(filepath, 'r') as fp: 
        lines = fp.readlines()
    return {i+1: line.strip('\n') for i, line in enumerate(lines)}

def save_dic(obj: dict, filepath: Path):
    with open(filepath, 'wb') as fo: 
        pickle.dump(obj, fo)

if __name__ == "__main__":
    
    # test to check if the whole dataset is read correctly
    train_images = read_all_images(TRAIN_DATA_PATH)
    test_images = read_all_images(TEST_DATA_PATH)
    print(train_images.shape)

    train_labels = read_labels(TRAIN_LABEL_PATH)
    test_labels = read_labels(TEST_LABEL_PATH)
    print(train_labels.shape)
    print(set(train_labels))

    classes = read_class_names(DATA_DIR / 'class_names.txt') 

    save_dic(classes, OUTPUT_DIR / 'class_index')
    # save images to disk
    save_images(train_images, train_labels, OUTPUT_DIR / 'train')  
    save_images(test_images, test_labels, OUTPUT_DIR / 'test')  