from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from numpy import zeros
import numpy as np
from sklearn.cluster import KMeans

def ikmeans(km, X, min_improvement=.01):
    ''' incremental kmeans; split worst cluster based on the silhouette score '''
    K = len(km.cluster_centers_)
    
    labels = km.labels_
    # compute label distribution
    labels_ratio = np.histogram(labels,bins=len(set(labels)))[0]
    labels_ratio = np.array(labels_ratio, dtype=float)/len(labels)
    scores = metrics.silhouette_samples(X, labels, metric='euclidean')
    score = scores.mean()

    # measure global performance of each cluster
    k_score = zeros(K)
    for k in range(K):
        idx = np.where(labels==k)
        k_score[k] = scores[idx].mean()
    
    # identify the cluster to split where population higher then min_population_ratio
    idx = np.where(labels_ratio>0.01)[0]
    worst_score = k_score[idx].max()
    worst_idx = np.where(k_score==worst_score)[0]
    if len(worst_idx)>1:
        print("several worst k (%i)" %len(worst_idx))
    worst_k = worst_idx[0]
    print("worsk cluster -> %i" %(worst_k))

    # split worst cluster
    idx = np.where(labels==k)[0]
    X_k = X[idx]
    if len(X_k)<=2:
        print("not enought data point to split")
        return 
    skm = KMeans(n_clusters=2, random_state=1).fit(X_k)
    
    # measure improvement with the 2 new clusters
    ikm = KMeans(n_clusters=K+1, random_state=1)
    new_centers = np.array(km.cluster_centers_).tolist()
    new_centers.remove(new_centers[worst_k])
    [new_centers.append(center) for center in skm.cluster_centers_]
    ikm.cluster_centers_ = np.array(new_centers)
    ilabels = ikm.predict(X)
    ikm.labels_ = ilabels
    new_score = metrics.silhouette_score(X, ilabels, metric='euclidean')
    
    improvement = (score - new_score)/score

    if improvement > min_improvement:
        print("increase k (%2.2f%%)" %(improvement*100))
        ikmeans(ikm, X)
    else:
        print("improvement %s->%s = %2.2f%%" %(K, K+1, improvement*100))
        print("best k = %i" %K)
        return km

if __name__ == "__main__":

    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target

    K=3
    km = KMeans(n_clusters=K, random_state=1).fit(X)
    ikmeans(km, X)

    

