import cv2
import numpy as np 

class dense():
    def __init__(self, step_size=5, kp_size=1):
        self.step_size = step_size
        self.kp_size = kp_size 

    @classmethod
    def create(self, step_size=5, kp_size=1):
        self.step_size = step_size
        self.kp_size = kp_size
        return dense(step_size=step_size, kp_size=kp_size)
    
    def detect(self, img, mask=None):
        # Assume greyscale
        h, w = img.shape
        kps = [
            cv2.KeyPoint(x, y, self.kp_size) 
                for y in range(0, h, self.step_size)
                for x in range(0, w, self.step_size)
            ]
        return kps 


    def compute():
        raise NotImplementedError('Dense keypoints is not a descriptor') 


class DenseHistDescriptor:
    def __init__(self, step_size=5, patch_size=32, grid=(4,4), bins=16):
        """
        Detector‑agnostic local intensity histogram descriptor on a dense grid.

        Args:
            step_size (int): spacing in pixels between keypoint centers.
            patch_size (int): width/height of square patch around each keypoint.
            grid (tuple): number of cells (rows, cols) per patch.
            bins (int): number of histogram bins per cell.
        """
        self.step_size = step_size
        self.patch_size = patch_size
        self.grid = grid
        self.bins = bins
        self.half = patch_size // 2

    @classmethod
    def create(cls, step_size=5, patch_size=32, grid=(4,4), bins=16):
        return cls(step_size=step_size, patch_size=patch_size, grid=grid, bins=bins)

    def detect(self, img, mask=None):
        """Generate a dense grid of keypoints over the image."""
        h, w = img.shape[:2]
        kps = []
        for y in range(self.half, h - self.half, self.step_size):
            for x in range(self.half, w - self.half, self.step_size):
                kps.append(cv2.KeyPoint(x, y, self.patch_size))
        return kps

    def compute(self, img_gray, keypoints, mask=None):
        """Compute local histograms around each keypoint."""
        descriptors = []
        cell_h = self.patch_size // self.grid[0]
        cell_w = self.patch_size // self.grid[1]

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            # extract patch
            patch = img_gray[y-self.half:y+self.half, x-self.half:x+self.half]
            if patch.shape != (self.patch_size, self.patch_size):
                continue

            hist_vals = []
            for i in range(self.grid[0]):
                for j in range(self.grid[1]):
                    cell = patch[
                        i*cell_h:(i+1)*cell_h,
                        j*cell_w:(j+1)*cell_w
                    ]
                    hist = cv2.calcHist([cell], [0], None, [self.bins], [0, 256])
                    # normalize each cell histogram to unit L1 norm
                    hist = cv2.normalize(hist, hist, alpha=1, beta=0, norm_type=cv2.NORM_L1)
                    hist_vals.extend(hist.flatten())

            descriptors.append(hist_vals)

        if len(descriptors) == 0:
            return [], np.array([], dtype=np.float32)
        return keypoints, np.array(descriptors, dtype=np.float32)
    
_DETECTOR_CREATES = {
    name.replace("_create", "").lower(): getattr(cv2, name.replace("_create", ""))
    for name in dir(cv2)
    if name.lower().endswith("_create")
}

_DETECTOR_CREATES['dense'] = dense 
_DETECTOR_CREATES['dense_hist'] = DenseHistDescriptor
def get_cv2_detector(name: str):
    func = _DETECTOR_CREATES.get(name.lower())
    if func is None: 
        msg = f"Available Detectors:\n----\n{"\n".join(list(_DETECTOR_CREATES))}"
        raise ValueError(f"No OpenCV detector named {name!r}.\n{str(msg)}")
    
    return func
