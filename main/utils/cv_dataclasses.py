import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path 

@dataclass
class ProcessedImages:
    norm_imgs: list[np.ndarray] = field(default_factory=list)
    keypoints: list[list[cv2.KeyPoint]] = field(default_factory=list)
    descriptors:  list[np.ndarray] = field(default_factory=list)
    descriptor_type: str  = field(default_factory=str)
    train: bool = field(default_factory=str)

@dataclass
class Images: 
    filepath: list[Path] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    train: bool = field(default_factory=bool)
    directory: str = field(default_factory=str)
    imgs: list[np.ndarray] = field(default_factory=list)

    def __add__(self, other: 'Images') -> 'Images':
        if not self.directory:
            self.directory = other.directory
        try:
            assert self.directory == other.directory 
        except(AssertionError): 
            print(self.directory, other.directory)
            raise AssertionError('Files must come from same root directory')
        
        return Images(
            filepath = self.filepath + other.filepath, 
            labels = self.labels + other.labels, 
            train = other.train,
            directory = self.directory,
            imgs = self.imgs + other.imgs,
        )
    
    @classmethod 
    def concat(cls, parts: list['Images']) -> 'Images':
        result = cls()
        for part in parts: 
            result = result + part 
        return result 