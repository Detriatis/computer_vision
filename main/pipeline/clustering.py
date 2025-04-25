import numpy as np
import matplotlib.pyplot as plt
from main.utils.cv_dataclasses import ProcessedImages

def k_means_clustering(vectors, k, max_iters=1000) -> tuple[np.ndarray, dict]:
    n_samples, n_features = vectors.shape
    
    initial_idxs = np.random.choice(n_samples, size=k, replace=False)
    means = vectors[initial_idxs].astype(float)

    for i in range(max_iters):
        if i % 5 == 0:
            print('At iteration', i) 
        distances = np.linalg.norm(vectors[:, None, :] - means[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)

        new_means = np.zeros_like(means)
        for idx in range(k):
            assigned = vectors[assignments == idx]
            if len(assigned) == 0:
                new_means[idx] = vectors[np.random.randint(0, n_samples)]
            else:
                new_means[idx] = assigned.mean(axis=0)

        if np.allclose(means, new_means):
            print('Converged, breaking early at iteration', i)
            break
        means = new_means

    clusters = {i: vectors[assignments == i] for i in range(k)}
    return means, clusters

def bag_of_words(imgs: ProcessedImages, k, max_iters, normalise: bool=True, means = None) -> tuple[np.ndarray, np.ndarray]:
    descs = imgs.descriptors
    n_images = len(imgs.descriptors)

    full_descs = np.concat(descs)
    if imgs.train: 
        print('Beginning Clustering')
        means, _ = k_means_clustering(full_descs, k, max_iters)
    else:
        try:
            assert means is not None
        except AssertionError:
            raise AssertionError('If providing test data please provide centroids')

    bows = np.zeros((n_images, k))
    
    for idx, d in enumerate(descs): 
        vecs = np.stack(d)
        distances = np.linalg.norm(vecs[:, None, :] - means[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        for i in assignments:
            bows[idx, i] += 1

    if normalise:
        row_sums = bows.sum(axis=1, keepdims=True)
        nonzero = row_sums.squeeze() > 0
        bows[nonzero] /= row_sums[nonzero]
    
    return bows, means
