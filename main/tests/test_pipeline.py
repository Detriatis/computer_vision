from main.utils.image_loading import load_image
import matplotlib.pyplot as plt
import cv2

TEST_IMAGE = "/home/REDACTED/Projects/manchester/computer_vision/datasets/processed/cifar-10/train/1/aeroplane_s_000004.png"

if __name__ == "__main__":
    img = load_image(TEST_IMAGE) 

    detector = cv2.SIFT_create()
    step_size = 5
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0] + (step_size), step_size)
                                        for x in range(0, img.shape[1] + (step_size), step_size)]
    kps = detector.detect(img, None)
    kps, des = detector.compute(img, kps)
    img2 = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
    plt.imshow(img2)
    plt.show()