from main.pipeline.clustering import k_means_clustering
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    vecs = np.random.randint(0, 255, (1000, 2))
    k = 3
    col = {
        0: 'green',
        1: 'blue',
        2: 'black',
    }
    means, clusters = k_means_clustering(vecs, k)
    for i in range(k):
        c_vecs = clusters[i]
        mean = means[i]
        c_vecs = np.concat(c_vecs, axis=0).reshape(-1, 2)
        plt.scatter(c_vecs[:, 0], c_vecs[:, 1], c=col[i])
        plt.scatter(mean[0], mean[1], c='red')

    plt.show()
