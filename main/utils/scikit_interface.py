from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, accuracy_score, auc, average_precision_score, f1_score, confusion_matrix, recall_score
from sklearn.base import ClassifierMixin, BaseEstimator
from typing import Protocol, Sequence, Optional, Any, Dict, cast, TypeVar, Union

import torch
from torch import nn 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer 
from torch.nn.modules.loss import _Loss
from torch.utils.data import TensorDataset, DataLoader

from main.utils.cv_dataclasses import ImagesDataset

import logging 
import numpy as np

class MetricFunction(Protocol):
    def __call__(
            self, 
            y_true: Sequence[Any],
            y_pred: Sequence[Any],
            *,
            labels: Optional[Sequence[Any]] = None,
            sample_weight: Optional[Sequence[float]] = None,
            **kwargs: Any,
    ) -> float: ... 

_CLASSIFIERS: Dict[str, ClassifierMixin] = {
    'svc': SVC,
    'rf': RandomForestClassifier,
    'knn': KNeighborsClassifier
}

U = TypeVar("U", bound=Union[BaseEstimator, ClassifierMixin])

_METRICS: Dict[str, MetricFunction]= {
    'ce': log_loss,
    'accuracy': accuracy_score,
    'precision': average_precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'auc': auc,
    'confmatrix': confusion_matrix
}

def get_sklearn_classifier(name: str) -> U:
    func = _CLASSIFIERS.get(name.lower())
    if func is None: 
        msg = f"Available Classifiers:\n----\n{"\n".join(list(_CLASSIFIERS))}"
        raise ValueError(f"No scikit classifiers named {name!r}.\n{str(msg)}")
    return func 

def get_sklearn_metric(name: str) -> MetricFunction:
    func = _METRICS.get(name.lower())
    if func is None: 
        msg = f"Available Metrics:\n----\n{"\n".join(list(_METRICS))}"
        raise ValueError(f"No scikit metric named {name!r}.\n{str(msg)}")
       
    return cast(MetricFunction, func)

from sklearn.base import BaseEstimator, TransformerMixin
import cv2
from main.utils.cv2_interface import get_cv2_detector
from main.pipeline.clustering import bag_of_words

class ImageNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm_type = cv2.NORM_MINMAX, alpha=0, beta=255, verbocity=0):
        self.norm_type = norm_type
        self.alpha = alpha
        self.beta = beta
        self.verbocity = verbocity 
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.verbocity: 
            print(f'Transforming data with normaliser {self.norm_type}')
        out = []
        for img in X:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.shape != (96, 96):
                img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
            if self.norm_type == cv2.NORM_MINMAX:
                norm_img = cv2.normalize(img, None, alpha=self.alpha, beta=self.beta, norm_type=self.norm_type)
            elif self.norm_type == cv2.NORM_L1:
                norm_img = cv2.normalize(img, None, alpha=self.alpha, beta=self.beta, norm_type=self.norm_type, dtype=cv2.CV_32F)
            elif self.norm_type == cv2.NORM_L2:
                norm_img = cv2.normalize(img, None, alpha=self.alpha, beta=self.beta, norm_type=self.norm_type, dtype=cv2.CV_32F)
            elif self.norm_type == 'eq_hist':
                norm_img = cv2.equalizeHist(img, None)

            try:  
                out.append(norm_img)
            except: 
                raise TypeError(f'{self.norm_type}')
        if self.verbocity > 1:
            print('The transformed image', out[0])
        return out
 
class DescriptorExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, kp_algo='SIFT', desc_algo='SIFT', verbocity=0, **detector_params):
        self.kp_algo = kp_algo 
        self.desc_algo = desc_algo 
        self.detector_params = detector_params
        self.verbocity = verbocity

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        if self.kp_algo.lower() == 'none':
            return X
        detector = get_cv2_detector(self.kp_algo).create(**self.detector_params)
        descriptor = get_cv2_detector(self.desc_algo).create() 
        if self.verbocity:
            print(f'Transforming data with {self.kp_algo} kps and {self.desc_algo} descriptors')
        
        kps, descs = [], []
        for i, img in enumerate(X): 
            kp = detector.detect(img, mask=None)
            if not kp or len(kp) == 0:
                print(f'SIFT failed on index {i}') 
            kp, desc = descriptor.compute(img, kp) 
            
            kps.append(kp)
            descs.append(desc) 
        return descs
    
class BoWTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=100, max_iters=100, batch=1024, normalise: bool = True):
        self.k = k
        self.max_iters = max_iters
        self.batch = batch
        self.normalise = normalise

    def fit(self, X, y=None):
             
        kmeans = MiniBatchKMeans(
            n_clusters = self.k,
            max_iter = self.max_iters, 
            batch_size  = self.batch,
            random_state=0
        )
        print('Beginning Clustering')
        self.kmeans = kmeans.fit(X)
    
        return self 
    
    def transform(self, X):
        bows = np.zeros((len(X), self.k), dtype=np.float32)

        for i, D in enumerate(X):
            if len(D) == 0:
                print(f'Image {i} has no descriptors')
                continue
            labels = self.kmeans.predict(D)
            for lbl in labels:
                bows[i, lbl] += 1
        if self.normalise:
            row_sums = bows.sum(axis=1, keepdims=True)
            nonzero = row_sums.squeeze() > 0
            bows[nonzero] /= row_sums[nonzero]
        
        return bows

class SKlearnPyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, net_cls: nn.Module, convs: int = 1,
                 optimizer_cls = optim.SGD,
                 criterion=nn.CrossEntropyLoss(),
                 epochs=10, batch_size=32, lr=1e-3, momentum=0.9, device=0):
        
        self.net_cls = net_cls
        self.convs = convs 
        self.optimizer_cls = optimizer_cls
        self.momentum = momentum
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps'
        else: 
            self.device = device
    
    def _build(self):

        self.model = self.net_cls(self.convs).to(self.device)
        self.optimizer = self.optimizer_cls(
            self.model.parameters(),
            lr = self.lr,
            momentum=self.momentum
        )
    
    def fit(self, X, y): 
        self._build()
        X_tensor = torch.tensor(X).permute(0, 3, 1, 2).float().to(self.device)
        y_tensor = torch.tensor(y).long().to(self.device)
        ds = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):

            if epoch % 2 == 0:
                print('Epoch: ', epoch)
            for xb, yb in loader:
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb) 
                loss.backward()
                self.optimizer.step()
        
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X).permute(0, 3, 1, 2).float().to(self.device)
        loader = DataLoader(X_tensor, batch_size=self.batch_size)
        all_preds = []
        with torch.no_grad():
            for xb in loader:
                out = self.model(xb)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds)

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X).permute(0, 3, 1, 2).float().to(self.device)
        loader = DataLoader(X_tensor, batch_size=self.batch_size)
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                out = self.model(xb)
                probs = F.softmax(out, dim=1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X)) 