#!/usr/bin/env python
""" Simplify clustering visualisation using scikit-learn
    example: python visu.py -m pca (visualize PCA dim reduction on news default dataset)
             python visu.py -d default -m pca -f data.tsv --no-text 
"""

import sys
import traceback
import matplotlib 
import warnings
# load interface; return data_train, data_test that contains data & targets 
from load_default import load as load_default
from news import load_news

def filter_classes(X, Y, classes, remove):
    ''' remove classes points from X and Y '''
    print("FILTERING CLASSES")
    import numpy as np
    cidx=[]
    for c in np.array(classes,dtype=int):
        cidx.extend(np.where(Y==c)[0])
    if remove:
        idx=[i for i in range(len(Y)) if i not in cidx]
    else:
        idx=[i for i in range(len(Y)) if i in cidx]
    idx = np.array(idx)
    print('selecting %i/%i indexes' %(len(idx),len(X)))
    return X[idx], Y[idx]   

_load_map = {'default':load_default, 'news':load_news}

def add_loading_dataset_fct(name, fct):
    if name in _load_map:
        print("load name already used %s" %name)
        sys.exit(1)
    else:
        _load_map[name] = fct
 
def size_mb(docs):
    try:
        return sum(len(s.encode('utf-8')) for s in docs) / 1e6
    #return sum(len(s) for s in docs) / 1e6
    except:
        return 0

def main(args=None):
    args = args.split(' ') if isinstance(args, str) else args
    args = args or sys.argv[1:]
    import logging
    import numpy as np
    from optparse import OptionParser
    from time import time
    from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
    from sklearn.feature_selection import SelectKBest, chi2    
    import dim_reduction as dr

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    # parse commandline arguments
    op = OptionParser()
    
    op.add_option("--chi2_select", default=-1,
                  action="store", type="int", dest="select_chi2",
                  help="Select some number of features using a chi-squared test; all set -1")
    op.add_option('-f',"--filename", default="data.tsv",
		  dest="fname", help="data filename")
    op.add_option("-d", "--dataset", default='news',
                  dest="dataset",
                  help="dataset to load (%s)" %list(_load_map.keys()))
    op.add_option('-n', "--n_features",
                  action="store", type=int, default=1000,
                  help="n_features when using the hashing vectorizer.")
    op.add_option("--use_hashing", default=False,
                  action="store_true",
                  help="Use a hashing vectorizer.")
    op.add_option("--hack", default=False,
                  action="store_true", dest="hack",
                  help="use test instead on train to speedup process") 
    op.add_option("--no-text", default=True,
                  action="store_false", dest="text",
                  help="features are not text (default = text)")
    op.add_option("--class_sample", default=2, type=int, 
                  dest="n_sample_by_class",
                  help="show only [%default%] sample by class")
    op.add_option("--lnob", default=True, action='store_true',
                  dest='legend_outside_box',
                  help="legend not outside of the box")
    op.add_option("--legend", default=False, action='store_true',
                  dest='enable_legend_picking', 
                  help='set legend picking not points')  
    op.add_option("--noX", default=False, action='store_true',
                  dest='nox', 
                  help="if you just want to generate graph and don't have acess to the X server ") 
    op.add_option("-m","--methods", default=dr.METHODS,
                  dest="methods",
                  help="dimension reduction method to try (split by ','); default = %s" %dr.METHODS)
    op.add_option("-e", dest='exclude', default=None,
                  help="exclude class (separarated by ,)")
    op.add_option("-o", dest='only', default=None,
                  help="include only class (separarated by ,)")
    op.add_option("-v", dest='verbose', default=False, action='store_true',
                  help="verbose")

    (opts, args) = op.parse_args(args)
    
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    if opts.nox:
        matplotlib.use('Agg')
    # warning: pylab should be import after call to matplotlib.use(...)
    import pylab
    
    # load data 
    data_train, data_test, legend_labels = _load_map[opts.dataset](opts.fname)

    if opts.hack:
        print("hack: working on test dataset")
        data_train = data_test
        opts.dataset+='_test'
        
    if opts.verbose:
        print("----------example data loaded--------------")
        print("data:" ,data_train.data[0].strip())
        print("target:", data_train.target[0])
        print("-------------------------------------------")
    y_train, y_test = data_train.target, data_test.target

    data_train_size_mb = size_mb(data_train.data)
    data_test_size_mb = size_mb(data_test.data)

    print(("%d documents - %0.3fMB (training set)" % (
            len(data_train.data), data_train_size_mb)))
    print(("%d documents - %0.3fMB (test set)" % (
            len(data_test.data), data_test_size_mb)))


    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    if not opts.text:
        print("std features")
        X_train = np.array(data_train.data, ndmin=2)
        features_names = data_train.features
        
    else: # its text features dood
        print("features are extracted from text -> words vectorization is required, hey Samu!")
        if opts.use_hashing:
            print(("Use feature hashing %s" %opts.n_features))
            vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                           n_features=opts.n_features)
            X_train = vectorizer.transform(data_train.data)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english')
            # mapping from integer feature name to original token string
            X_train = vectorizer.fit_transform(data_train.data)
            feature_names = vectorizer.get_feature_names()

    if opts.verbose:
        print("----------example data transformed--------------")
        print("data:" ,X_train[0])
        print("target:", y_train[0])
        print("-------------------------------------------")

    duration = time() - t0
    print(("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration)))
    print(("n_samples: %d, n_features: %d" % X_train.shape))
    print()
    
    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time()
    if not opts.text:
        X_test = np.array(data_test.data, ndmin=2)
    else:
        X_test = vectorizer.transform(data_test.data)
    duration = time() - t0
    print(("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration)))
    print(("n_samples: %d, n_features: %d" % X_test.shape))
    print()

    if opts.select_chi2!=-1:
        print(("Extracting %d best features by a chi-squared test" %
              opts.select_chi2))
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        print("data:", X_train[0])
        print("target", y_train[0])
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        print(("done in %fs" % (time() - t0)))
        print()

    X = X_train.todense() if "todense" in dir(X_train) else X_train
    X_test = X_test.todense() if "todense" in dir(X_test) else X_test
    print("data shape: (%i,%i)" %(X.shape))

    if opts.only:
        idx = opts.only.split(',')
        X, y_train = filter_classes(X, y_train, idx, False)
        X_test, y_test = filter_classes(X_test, y_test, idx, False)

    if opts.exclude:
        idx = opts.exclude.split(',')
        X, y_train = filter_classes(X, y_train, idx, True)
        X_test, y_test = filter_classes(X_test, y_test, idx, True)


    # run all dim reduction algo
    for method in opts.methods.split(','):
        t0 = time()
        try:
            resdr = dr.dim_reduce(method, X=X, Y=y_train)
            if resdr == None:
                continue
            trans,X_trans,title = resdr
            print(('Projecting {} on test set'.format(method)))
            if hasattr(trans,"transform"):
                X_trans_test = trans.transform(X_test)
            elif hasattr(trans,"fit_transform"):
                warnings.warn("the method as no transform (fallback to fit_transform", Warning) 
                X_trans_test = trans.fit_transform(X_test)
            title = "%s (time %.2fs)" %(title,(time() - t0))
            print(('Rendering plot {}'.format(title)))
            has_plot = dr.plot_embedding(X=X_trans_test, Y=y_test, title=title, 
                              n_sample_by_class=opts.n_sample_by_class,
                              source=data_test.data, 
                              legend_outside_box=opts.legend_outside_box,
                              enable_legend_picking=opts.enable_legend_picking,
                              legend_labels=legend_labels)
            if has_plot:
                fname = "%s_%s.png" %(opts.dataset, method)
                print("saving %s" %fname)
                pylab.savefig(fname, bbox_inches=0)
            else:
                print('Nothing to plot.')

        except Exception as ex:
            print(method, ex)
            print(traceback.format_exc())
        
    pylab.show()

if __name__ == '__main__':
    main()
