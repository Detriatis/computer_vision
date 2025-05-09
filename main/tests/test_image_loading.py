from main.utils.image_loading import load_image, load_labeled_images, load_directory_images
import cv2
from pathlib import Path

TESTFILE = "/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/mammals/train/2/red_panda-0001.jpg"
TESTCLASSDIR = Path("/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/cifar-10/train/1")
TESTIMGDIR = Path("/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/cifar-10/train/")

def test_image_loading(img = None):
    img = load_image(TESTFILE)

    if img is None: 
        raise RuntimeError("Could not load image!")
    
    cv2.namedWindow('Test Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Test Image', img)
    print('Press a key to close the window')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    return 

def test_load_labeled_images(): 
    imgs = load_labeled_images(TESTCLASSDIR)
    print(f'Loaded {len(imgs.imgs)} images from {TESTCLASSDIR.parent.name}')
    assert len(imgs.imgs) == 5000 

def test_load_directory_images():
    imgs = load_directory_images(TESTIMGDIR)
    print(f'Loaded {len(set(imgs.labels))} classes')
    print('Training:', imgs.train)
    # Checking standard cifar-10 image number 
    print(len(imgs.labels))
    assert len(imgs.labels) == 50000
    assert len(set(imgs.labels)) == 10


if __name__ == '__main__':
    test_image_loading()
    test_load_labeled_images()
    test_load_directory_images()
