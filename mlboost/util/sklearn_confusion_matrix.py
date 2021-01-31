''' example of confusion matrix integration in sklearn '''




import logging
import numpy as np
import sys
from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


k = 10 

###############################################################################
# Load some categories from the training set

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    ]

print("Loading 20 newsgroups dataset for categories:")

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)

y_train = data_train.target

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)


duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" %X_train.shape)
print()

clf = KNeighborsClassifier(n_neighbors=10)

from mlboost.util.confusion_matrix import ConfMatrix

clf.fit(X_train, y_train)
pred = clf.predict(X_train)

labels = list(set(y_train))
labels.sort()

cm = ConfMatrix(metrics.confusion_matrix(y_train, pred), labels)
cm.save_matrix('conf_matrix.p')
cm.get_classification()
cm.gen_conf_matrix('conf_matrix')
cm.gen_highlights('conf_matrix_highlights')

