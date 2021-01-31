from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from numpy import zeros
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

def compute_improvement(ref, new):
    ''' improvement is when the sihouette coefficient increase '''
    return float(new-ref)/ref

def compute_label_dist(labels):
    dist=dict([(l,0.0) for l in set(labels)])
    for l in labels:
        dist[l]+=1
    for l in set(labels): 
        dist[l]/=len(labels)*100
    return dist

def ikmeans(km, X, min_improvement=.01, n_partial_fit=10, verbose=False):
    ''' incremental kmeans; split worst cluster based on the silhouette score 
     higher Silhouette Coefficient score relates to a model with better defined clusters
'''
    def get_k_score(km):
        # measure global performance of each cluster
        k_score = zeros(km.n_clusters)
        scores = metrics.silhouette_samples(X, km.labels_, metric='euclidean')
        for k in range(km.n_clusters):
            idx = np.where(km.labels_==k)
            k_score[k] = scores[idx].mean()
        return k_score, scores

    K = km.n_clusters
    
    labels = km.labels_
    # compute label ratio distribution
    labels_dist = np.histogram(labels,bins=len(set(labels)))[0]
    labels_dist = np.array(labels_dist, dtype=float)/len(labels)*100
    
    #labels_ratio = compute_label_dist(labels)

    k_score, scores = get_k_score(km)
    score = scores.mean()

    # identify the cluster to split where population higher then min_population_ratio
    idx = np.where(labels_dist>1)[0]
    worst_score = k_score[idx].min()
    worst_idx = np.where(k_score==worst_score)[0]
    if len(worst_idx)>1:
        print("several worst k (%i)" %len(worst_idx))
    worst_k = worst_idx[0]
    print("worsk cluster -> %i (%.2f%%)" %(worst_k, labels_dist[worst_idx]))

    # split worst cluster
    idx = np.where(labels==worst_k)[0]
    X_k = X[idx]
    if len(X_k)<=2:
        print("not enought data point to split")
        return 
    # generate 2 new cluster on works k 
    worstk_km = KMeans(n_clusters=2).fit(X_k)
    
    # measure improvement with the 2 new clusters
    ikm = MiniBatchKMeans(n_clusters=K+1)
    new_centers = np.array(km.cluster_centers_).tolist()
    new_centers.remove(new_centers[worst_k])
    [new_centers.append(center) for center in worstk_km.cluster_centers_]
    ikm.cluster_centers_ = np.array(new_centers)

    # readjust means 
    if n_partial_fit:
        for i in range(n_partial_fit):
            ikm.partial_fit(X)

    ilabels = ikm.predict(X)
    ikm.labels_=ilabels

    if verbose:
        print("centers")
        print(km.cluster_centers_)
        print(ikm.cluster_centers_)
        new_labels_dist = np.histogram(ilabels,bins=len(set(ilabels)))[0]
        new_labels_dist = np.array(new_labels_dist, dtype=float)/len(labels)*100
        print("old labels dist:",labels_dist)
        print("new labels dist:",new_labels_dist)
    
    new_score = metrics.silhouette_score(X, ilabels, metric='euclidean')

    improvement = compute_improvement(score, new_score)

    if improvement > min_improvement:
        print("increase k to %i (improvement = %2.2f%%; %.3f->%.3f))" %(K+1, improvement*100, score, new_score))
        return ikmeans(ikm, X)# TODO
    else:
        print("improvement %s->%s = %2.2f%% (%.3f -> %.3f)" %(K, K+1, improvement*100, score, new_score))
        print("best k = %i (score = %.3f)" %(K, new_score))
        return km, new_score

if __name__ == "__main__":

    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    print("n classes", len(set(dataset.target)))
 
    K=2
    skm = MiniBatchKMeans(n_clusters=K)
    skm.fit(X)
    best_km, best_score = ikmeans(skm, X)
    score_ref = metrics.silhouette_score(X, skm.labels_, metric='euclidean')
    print("ref score: %.2f (k at start)" %(score_ref))
    print("ikmeans improvement %2.2f%% (%.2f->%.2f)" %(compute_improvement(score_ref, best_score)*100, best_score, score_ref)) 

