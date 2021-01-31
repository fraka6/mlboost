# example of clustering visualisation using scikit-learn

METHODS = 'random,pca,lda,isomap,lle,mlle,hlle,ltsa,mds,rtree,spectral,tsne'

import numpy as np
import traceback
from time import time
from random import choice, random
from sklearn import manifold, decomposition, ensemble, random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda


from mlboost.core.pphisto import SortHistogram, Histogram


def plot_embedding(X, Y, title=None, n_sample_by_class=2, min_freq=100, 
                   sampling=.1, legend_outside_box=True, enable_legend_picking=False, source=None,
                   legend_labels=None):
    '''Scale and visualize the embedding vectors'''

    # Make sure there is no line with NaN value
    nans = np.isnan(X)
    retained = np.invert(np.any(nans, 1))
    retained_idx = np.where(retained)[0]
    X = X[retained_idx, :]
    if len(X) == 0:
        # Nothing to plot
        return False

    import pylab as pl
    y = np.array(Y,dtype=str)    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    def get_point_info(x,y):
        for i, row in enumerate(X):
            if row[0]==x and row[1]==y:
                msg = "[{x},{y}] = row {i}".format(x=x,y=y,i=i)
                if source:
                    msg+='-> class : {c}\n{content}'.format(c=Y[i],content=source[i])
                print(msg)

    fig = pl.figure()
    ax = pl.subplot(111)

    lines = [] # this is used to enable picking on legend (see below)

    # show untagged data points
    tidx = np.where(y=='?')[0]
    if len(tidx):
        sX = X[tidx,:]
        sX = sX[[i for i in range(len(tidx)) if random()<sampling],:]
        if len(sX):
            lines.append(ax.plot(sX[:,0],sX[:,1],'.', label='?', picker=True)[0])
    
    # show some tagged samples
    tidx = np.where(y!='?')[0]
    
    dist = Histogram(y[tidx])
    sdist = SortHistogram(dist, False, True)
    classes = [label for label, count in sdist if count>min_freq]
    classes.sort()

    for i, cl in enumerate(classes):
        legend_name = legend_labels[i] if legend_labels else cl
        tidx = np.where(y==str(cl))[0]
        cX = X[tidx,:]
        color = float(classes.index(cl))/len(classes) 
        lines.append(ax.plot(cX[:,0],cX[:,1],'.', color=pl.cm.Set1(color), label=legend_name, picker=True)[0])
        
        # show some label on the graph
        for i in range(n_sample_by_class):
            if len(cX)>0:
                idx = choice(list(range(len(cX))))
                pl.text(cX[idx, 0], cX[idx, 1], str(cl),
                        color=pl.cm.Set1(float(classes.index(cl))/len(classes)),
                        fontdict={'weight': 'bold', 'size': 9})
    
          
    if legend_outside_box:
        # Shink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        leg = pl.legend()
    
    if enable_legend_picking:
        # enable picking on the legend 
        lined = dict()

        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)  # 5 pts tolerance
            lined[legline] = origline

        def onpick(event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', onpick)
    else:
        # point picking (with data)
        from matplotlib.lines import Line2D
        def onpick1(event):
            if isinstance(event.artist, Line2D):
                thisline = event.artist
                xdata = thisline.get_xdata()
                ydata = thisline.get_ydata()
                ind = event.ind
                x=np.take(xdata, ind)
                y=np.take(ydata, ind)
                n=len(x)+1
                xy = list(zip(x,y))
                print(('%i points: ' %n))
                for i,(x,y) in enumerate(xy):
                    print(("#%i)" %(i+1)))
                    get_point_info(x, y)
                
        fig.canvas.mpl_connect('pick_event', onpick1)
    
    if title is not None:
        pl.title(title)

    pl.xlabel('dim 1')
    pl.ylabel('dim 2')
    return True

def randomp(X, dim=2, **kargs):
    '''Random 2D projection using a random unitary matrix'''
    print("Computing random projection")
    try:
        rp = random_projection.SparseRandomProjection(n_components=dim, random_state=42)
        X_projected = rp.fit_transform(X)
        return rp, X_projected, "Random Projection"
    except Exception as e:
        traceback.print_exc()
        
def pca(X, dim=2, **kargs):
    '''Projection on to the first 2 principal components'''
    
    print("Computing PCA projection")
    try:
        pca = decomposition.PCA(n_components=dim)
        X_pca = pca.fit_transform(X)
        return pca, X_pca, "Principal Components projection" 
    except Exception as e:
        traceback.print_exc()

class ldaModel(object):
    def __init__(self, model):
        self.model = model

    def getInvertible(self, X):
        X2 = X.copy()
        X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
        return X2

    def fit(self, X, y=None):
        X2 = self.getInvertible(X)
        return self.model.fit(X2, y)

    def fit_transform(self, X, y=None):
        X2 = self.getInvertible(X)
        return self.model.fit_transform(X2, y)

    def transform(self, X):
        X2 = self.getInvertible(X)
        return self.model.transform(X2)

    def predict(self, X):
        X2 = self.getInvertible(X)
        return self.model.predict(X2)

def lda(X, Y, dim=2, **kargs):
    '''Projection on to the first 2 linear discriminant components'''
    print("Computing LDA projection")
    try:
        ldatr = ldaModel(lda.LDA(n_components=dim))
        X_lda = ldatr.fit_transform(X, Y)
        return ldatr, X_lda, "Linear Discriminant projection of the features"
    except Exception as e:
        traceback.print_exc()
        
def isomap(X, dim=2, n_neighbors=30, **kargs):
    '''Isomap projection of the dataset'''
    print("Computing Isomap embedding")
    try:
        isomap = manifold.Isomap(n_neighbors, n_components=dim)
        X_iso = isomap.fit_transform(X)
        print("Done.")
        return isomap, X_iso, "Isomap projection of the features"
    except Exception as e:
        traceback.print_exc()

def lle(X, dim=2, n_neighbors=30, **kargs):
    '''Locally linear embedding of the dataset'''
    print("Computing LLE embedding")
    methods = ['standard', 'ltsa', 'hessian', 'modified']
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                          eigen_solver='auto',
                                          method='standard')
    try:
        X_lle = clf.fit_transform(X)
        print(("Done. Reconstruction error: %g" % clf.reconstruction_error_))
        return clf, X_lle, "Locally Linear Embedding of the features"
        
    except Exception as e:
        traceback.print_exc()

def mlle(X, dim=2, n_neighbors=30, **kargs):
    '''Modified Locally linear embedding of the dataset'''
    print("Computing modified LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                          method='modified')
    try:
        X_mlle = clf.fit_transform(X)
        print(("Done. Reconstruction error: %g" % clf.reconstruction_error_))
        return clf, X_mlle, "Modified Locally Linear Embedding of the features"
    except Exception as e:
        traceback.print_exc()

def hlle(X, dim=2, n_neighbors=30, **kargs):
    '''HLLE embedding of the dataset'''
    print("Computing Hessian LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                          method='hessian')
    try:
        X_hlle = clf.fit_transform(X)
        print(("Done. Reconstruction error: %g" % clf.reconstruction_error_))
        return clf, X_hlle, "Hessian Locally Linear Embedding of the features"
    except Exception as e:
        traceback.print_exc()
        
def ltsa(X, dim=2, n_neighbors=30, **kargs):
    '''LTSA embedding of the dataset'''
    print("Computing LTSA embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                          method='ltsa')
    try:
        X_ltsa = clf.fit_transform(X)
        print(("Done. Reconstruction error: %g" % clf.reconstruction_error_))
        return clf, X_ltsa, "Local Tangent Space Alignment of the features"

    except Exception as e:
        traceback.print_exc()
        
def mds(X, dim=2, **kargs):
    '''MDS  embedding of the dataset'''
    print("Computing MDS embedding")
    clf = manifold.MDS(n_components=dim, n_init=1, max_iter=100)
    
    try:
        X_mds = clf.fit_transform(X)
        print(("Done. Stress: %f" % clf.stress_))
        return clf, X_mds, "MDS embedding of the features"
    except Exception as e:
        traceback.print_exc()
        
def rtree(X, dim=2, n_estimators=200, max_depth=5, **kargs):
    '''Random Trees embedding of the dataset'''
    print("Computing Totally Random Trees embedding")
    from sklearn.pipeline import Pipeline
    tr = Pipeline([
        ('hasher', ensemble.RandomTreesEmbedding(n_estimators=n_estimators, random_state=0,
                                           max_depth=max_depth)),
        ('pca', decomposition.PCA(n_components=dim))])

    try:
        X_reduced = tr.fit_transform(X)
        
        return tr, X_reduced, "Random forest embedding of the features"
    except Exception as e:
        traceback.print_exc()


def spectral(X, dim=2, **kargs):
    # Spectral embedding of the dataset
    # quite cool https://github.com/oreillymedia/t-SNE-tutorial
    print("Computing Spectral embedding")
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
                                          eigen_solver="arpack")
    try:
        X_se = embedder.fit_transform(X)
        
        return embedder, X_se, "Spectral embedding of the features"
        
    except Exception as e:
        traceback.print_exc()

def tsne(X, dim=2, **kargs):
    # t-sne embedding of the dataset
    print("Computing t-sne embedding")
    embedder = manifold.TSNE(n_components=dim, init='pca', random_state=0)
    try:
        X_se = embedder.fit_transform(X)
        
        return embedder, X_se, "t-sne embedding of the features"
        
    except Exception as e:
        traceback.print_exc()

_fct_map = {
    "random":randomp,
    "pca":pca,
    "lda":lda,
    "isomap":isomap,
    "lle":lle,
    "mlle":mlle,
    "hlle":hlle,
    "ltsa":ltsa,
    'tsne':tsne,
    # MDS is lacking a transform method, so cannot be used
    # to fit a transform on training set and project on the test set.
#    "mds":mds,
    "rtree":rtree,
    # SpectralEmbedding is lacking a transform method, so cannot be used
    # to fit a transform on training set, and project on the test set.
#    "spectral":spectral
    }

def fct_available():
    return list(_fct_map.keys())

def dim_reduce(algo_name, **kwargs):
    if algo_name not in _fct_map:
        print("warning: %s not available" %algo_name)
    else:
        return _fct_map[algo_name](**kwargs)
