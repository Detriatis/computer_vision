import cv2
import numpy as np
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
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
    
class ImagesDataset(Dataset):
    def __init__(self, X: np.array, y: np.array = None, transform=None, target_transform=None):
        self.images = X 
        self.transform = transform
        self.target_transform = target_transform
        self.labels = y

        if self.labels is not None:
            uniq = np.unique(y) 
            self.lbl2idx = {lbl: i for i, lbl in enumerate(sorted(uniq))}
            self.idx2lbl = {i: lbl for lbl, i in self.lbl2idx.items()}

            self.labels = np.vectorize(self.lbl2idx.get)(y).astype(np.int64)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)

        if self.labels is None:
            return image

        label = self.labels[idx]

        if self.target_transform:
            label = self.target_transform(label) 

        return image, label
