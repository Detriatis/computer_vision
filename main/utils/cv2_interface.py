import cv2

class dense():

    def create(self, img, step_size, kp_size):
        self.img = img 
        self.step_size = step_size
        self.kp_size = kp_size 

    def detect(self):
        # Assume greyscale
        h, w = self.img.shape
        kps = [
            cv2.KeyPoint(x, y, self.kp_size) 
                for y in range(0, h, self.step_size)
                for x in range(0, w, self.step_size)
            ]
        return kps 


    def compute():
        raise NotImplementedError('Dense keypoints is not a descriptor') 


_DETECTOR_CREATES = {
    name.replace("_create", "").lower(): getattr(cv2, name.replace("_create", ""))
    for name in dir(cv2)
    if name.lower().endswith("_create")
}

_DETECTOR_CREATES['dense'] = dense 

def get_cv2_detector(name: str, **kwargs):
    func = _DETECTOR_CREATES.get(name.lower())
    if func is None: 
        raise ValueError(f"No OpenCV detector named {name!r}.\n"
                         f"Available Detectors:\n----\n{"\n".join(list(_DETECTOR_CREATES))}")
    
    return func(**kwargs)
