#!/usr/bin/env python
''' Confusion matrix util class:
    - simple visualization
    - to precision and recall transformation
    - to classification erreur
    - generate confusion hightlights  
    - '''

import warnings
import numpy as np
from numpy import zeros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pickle
from collections import defaultdict
from csv import DictReader, DictWriter, QUOTE_NONE
import os.path
import itertools
import logging

class ConfMatrix:
    def __init__(self, matrix=None, labels=None):
        self.m = None
        # get by gen_conf_mapping()
        self.conf_mapping = None
        if matrix is not None:
            self.m = matrix
        if labels:
            self.labels = labels

    @staticmethod
    def load_matrix(fname):
        f = open(fname, "r")
        m = pickle.load(f)
        labels = pickle.load(f)
        f.close()
        return ConfMatrix(m, labels)

    @staticmethod
    def dict2confmatrix(d):
        labels = list(d.keys())
        n = len(labels)
        m = zeros((n,n))
        for i, key in enumerate(d):
            for label in d[key]:
                j = labels.index(label)
                m[i,j] = d[key][label]

        return ConfMatrix(m, labels)

    @staticmethod
    def gen_confusion_file(actual_fname, predicted_fname, class_field, output_dir):
        ''' return confusion mapping; key=class1->class2 to get all
            confusion tsv rows; full tsv files are expected''' 
        default = "N/A"
        f1 = open(actual_fname, 'rb')
        f2 = open(predicted_fname, 'rb')
        reader1=DictReader(f1, delimiter='\t', quoting=QUOTE_NONE, restval=default)
        reader2=DictReader(f2, delimiter='\t', quoting=QUOTE_NONE, restval=default)
        output_name = (os.path.basename(actual_fname)+"vs"+os.path.basename(predicted_fname)).replace(".tsv","")+".confusion.tsv"
        output = os.path.join(output_dir, output_name)
        fout = open(output, 'wb')
        writer = DictWriter(fout, reader1.fieldnames, delimiter="\t", quoting=QUOTE_NONE, quotechar=None)
        
        class_field2="%s2" %class_field
        writer.fieldnames.append(class_field2)
        writer.writeheader()    
        # keep confusion if 
        for row1,row2 in zip(reader1, reader2):
            
            if row1[class_field]!=row2[class_field]:
                row1[class_field2]=row2[class_field]
                writer.writerow(row1)
        print("generated: ",output)
        fout.close()
        f1.close()
        f2.close()

    @staticmethod
    def files2confmatrix(actual_fname, predicted_fname, class_field, constraint=None, 
                         force_predicted_idxs=False, gen_conf_file=True):
        ''' apply contraint (key=value) on class1 and class2 loading'''
        from sklearn import metrics
        f1 = open(actual_fname, 'rb')
        f2 = open(predicted_fname, 'rb')
        reader1=DictReader(f1, delimiter='\t', quoting=QUOTE_NONE)
        reader2=DictReader(f2, delimiter='\t', quoting=QUOTE_NONE)
        if constraint:
            key, value=constraint.split('=')
            if not force_predicted_idxs:
                actual = [row[class_field] for row in reader1 if row[key]==value]
                predicted = [row[class_field] for row in reader2 if row[key]==value]
            else: # force idxs (use actual idxs on predicted idxs)
                idx_actual = [(i,row[class_field]) for i,row in enumerate(reader1) if row[key]==value]
                actual = [c for i,c in idx_actual]
                idxs = {i: c for i, c in idx_actual}
                predicted = [row[class_field] for i,row in enumerate(reader2) if i in idxs]
        else:
            actual = [row[class_field] for row in reader1]
            predicted = [row[class_field] for row in reader2]
        f1.close()
        f2.close()

        print('Generating confusion matrix from labels')
        labels = sorted(set(actual).union(predicted))
        if len(actual)!=len(predicted):
            logging.warning("You should try --force-predicted-idx option")
            logging.error("something wrong, file don't have the same number of labels (%i!=%i)" %(len(actual),len(predicted)))
        return ConfMatrix(metrics.confusion_matrix(actual, predicted), labels)          

    def save_matrix(self, fname, labels=None):
        if '.' not in fname:
            fname+=".pickle"
        f = open(fname, "w")
        pickle.dump(self.m , f)
        pickle.dump(self.labels , f)
        print(("saving %s" %fname))
        f.close()

    def get_classification(self, labels=None, 
                           rm_labels=['tbd', '?','TBD']):
        labels = labels or self.labels
        idxs = [idx for idx, el in enumerate(labels) if el not in rm_labels]
        rm_idxs = [idx for idx, el in enumerate(labels) if el in rm_labels]
        if len(rm_idxs):
            print(("don't consider %s" %(','.join(["%s->%s" %(label, idx) for label, idx in zip(rm_labels, rm_idxs)]))))
        total = self.m[idxs,:].sum()
        target = 0.0
        for i in idxs:
            target+=self.m[i][i]
        print(("global precision: %i/%i=%2.2f" %(target, total, target/total)))
        return target/total

    def to_precision(self):
        precision = np.array(self.m, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            # Unavoidable divide by 0 if one element of sum is 0. Can only be avoided by
            # switching from vectorized to slower iterative form.
            precision/=self.m.sum(axis=0)    
        precision*=100
        return precision

    def to_recall(self):
        recall = np.array(self.m, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            recall/=self.m.sum(axis=1)[:,np.newaxis]
        recall*=100
        return recall

    @staticmethod
    def gen_matrix_figure(fig,
                          matrix, xlabels, ylabels,
                          threshold=5, factor=1,
                          title=None, xtitle=None, ytitle=None,
                          fontsize=5, grid=True, normfunc=None,
                          aspect='auto',
                          cmap='Greys',
                          fontfamily='serif'):
        xsize, ysize = matrix.shape[:2]
        ax = fig.add_subplot(111)
        #ax.set_aspect(2)

        # The matrix given by imshow is misleading.
        # First dimension corresponds to Y axis while second
        # dimension corresponds to X.
        # However, the given matrix has first dimension mapped to X and
        # second to Y. Matrix shape can be sizeX x sizeY or
        # sizeX x sizeY x 3|4 so we cannot just use .T
        axes = list(range(0, len(matrix.shape)))
        tr_axes = [axes[1], axes[0]] + axes[2:]
        tr_matrix = np.transpose(matrix, axes=tr_axes)
        # ensure nan don't screw stuff.
        # Doing this on tr_matrix makes the method better behaved, not modifying
        # its input argument unexpectedly.
        where_are_NaNs = np.isnan(tr_matrix)
        tr_matrix[where_are_NaNs] = 0
        res = ax.imshow(tr_matrix, cmap=cmap, 
                        interpolation='nearest', aspect=aspect,
                        norm=normfunc)

        if not np.isinf(threshold):
            # Iterating on the full matrix can be slow.
            # No need to do this if the threshold is infinite.
            for x in range(xsize):
                for y in range(ysize):
                    if np.any(abs(matrix[x][y])>threshold):
                        value = round(matrix[x][y]*factor)
                        color= 'white' if value >65 else 'black'
                        str_value = "%2.0f" %(value)
                        ax.annotate(str_value, xy=(x, y), 
                                    horizontalalignment='center',
                                    verticalalignment='center', color=color, size=fontsize,
                                    family=fontfamily)
                
        ax.set_xticks(list(range(xsize)))
        ax.set_xticklabels(xlabels, rotation='vertical',
                           fontsize=fontsize, family=fontfamily)
        ax.set_yticks(list(range(ysize)))
        ax.set_yticklabels(ylabels, fontsize=fontsize, family=fontfamily)

        fig.subplots_adjust(left=.2,right=0.98, top=0.95, bottom=0.35)
        if grid:
            ax.grid()
        cb = fig.colorbar(res)
        if title:
            ax.set_title(title, family=fontfamily)
        if xtitle:
            ax.set_xlabel(xtitle, family=fontfamily)
        if ytitle:
            ax.set_ylabel(ytitle, family=fontfamily)
        fig.tight_layout()

    def _gen_conf_matrix(self, fname, labels=None, title=None, threshold=5, 
                         factor=1, normalize="", xylabels=True, fontsize=5,
                         highlights=True, hthreshold=10, grid=True, output_dir='.'):
        fname = fname or "conf_matrix"
        
        labels = labels or self.labels
        matrix = self.m
        if normalize!='':
            fname+= "_%s" %normalize
        start_fname = str(fname)

        title = title or fname.replace('_',' ')
        #title += " %s" %normalize
        if xylabels:
            xtitle = 'Predicted classes'
            ytitle = 'Actual classes'
        else:
            xtitle = None
            ytitle = None
        matrix = np.array(matrix, dtype=float)
        
        if normalize=='recall':
            matrix=self.to_recall()

        elif normalize=='precision':
            matrix=self.to_precision()

        fig = plt.figure()
        ConfMatrix.gen_matrix_figure(fig, matrix, labels, labels, threshold, factor,
                          title, xtitle, ytitle, fontsize, grid)

        fname+=".png"
        fname = fname.replace(' ','_')
        #plt.tight_layout()
        if fontsize:
            plt.rc('font', family='serif', size=fontsize)
        fpath = os.path.join(output_dir, fname)
        #plt.savefig(fpath, format='png')# bbox_inches='tight')
        #plt.figure(figsize=(2,2)) # This increases resolution
        #plt.savefig('test.eps', format='eps', dpi=900)
        fig.savefig(fpath, format='png', dpi=900)
        plt.close(fig)
        print(("generated : %s" %fpath))
        if highlights:
            hname=start_fname.replace(' ','')+".highlight+%s.txt" % hthreshold
            hpath = os.path.join(output_dir, hname)
            with open(hpath, 'w') as fout:
                self.gen_highlights(fout, self.labels, hthreshold, matrix=matrix)
            print(("generated : %s" %hpath))

    def gen_conf_matrix(self, fname, labels=None, title=None, threshold=5, factor=1, fontsize=1,
                        highlights=True, hthreshold=10, grid=True, output_dir='.'):
        print('Generating plots and highlights')
        fname = fname or "conf_matrix"
        labels = labels or self.labels
        probably_normalized = self.m.max()<100
        self._gen_conf_matrix(fname, labels, title=title, threshold=threshold, factor=factor, 
                              fontsize=fontsize, hthreshold=hthreshold, grid=True, output_dir=output_dir)
        if not probably_normalized:
            self._gen_conf_matrix(fname, labels, title=title, threshold=threshold, factor=factor, 
                                  normalize='precision', fontsize=fontsize,hthreshold=hthreshold, grid=True,
                                  output_dir=output_dir)
            self._gen_conf_matrix(fname, labels, title=title, threshold=threshold, factor=factor, 
                                  normalize='recall', fontsize=fontsize, hthreshold=hthreshold, grid=True,
                                  output_dir=output_dir)
                
    def gen_highlights(self, fout, labels=None, val_threshold=75, matrix=None):
        from mlboost.core.pphisto import SortHistogram
        confusion_matrix = matrix if matrix is not None else self.m 
        labels = labels or self.labels

        for i1, label in enumerate(labels):
            i = 0
            dist = {}
            labels2 = list(labels)
            labels2.remove(label)
            for i2, label_2 in enumerate(labels2):
                if i2 >= i1:
                    i2 += 1
                if isinstance(confusion_matrix, list):
                    dist[label_2] = confusion_matrix[label][label_2]
                else: # assume numpy matrix
                    dist[label_2] = confusion_matrix[i1][i2]
                               
            sdist = SortHistogram(dist, False, True)
            
            confusions = ["%s %2.0f" %(key, value) for key, value in sdist if value >val_threshold 
                          and key not in('?','tbd')]
            if len(confusions)>0:
                if isinstance(confusion_matrix, list):
                    conf_title = '\n%s (%2.0f%%) -> ' %(label,confusion_matrix[label][label]*100)
                else: # assume numpy matrix
                    idx = i1
                    conf_title = '\n%s (%2.0f) -> ' %(label,confusion_matrix[idx][idx])
                fout.write(conf_title)
                fout.write(' | '.join(confusions))

if __name__ == "__main__":
    from optparse import OptionParser
    import pickle
    import sys

    parser = OptionParser(description=__doc__)
    
    parser.add_option("-f", dest="fname1", default = None, 
                      help="pickle matrix fname or tsv actual classification filename")
    parser.add_option("--fontsize", dest="fontsize", default = 8.0, 
                      help="font size to use")
    parser.add_option("-2", dest="fname2", default = None, 
                      help="second matrix fname (show diff) or tsv predicted classification filename")
    parser.add_option("--constraint", dest="constraint", default = None, 
                      help="constraint to apply on fname1 and fname 2: key=val")
    parser.add_option("-t", dest="title", default = None, 
                      help="title")
    parser.add_option("-M", dest="threshold", default = .05, type=float, 
                        help="min threshold")
    parser.add_option("-o", dest='output', default=None,
                        help="outputfile")
    parser.add_option("-c", dest='class_field', default=None,
                      help="classification  fieldname (if fname1 and fname2 are tsv files)")
    parser.add_option("-H", dest='hthreshold', default=10, type=float,
                        help="highlights threshold")
    parser.add_option("-v", dest='verbose', default=False, action='store_true',
                        help="activate verbose")
    parser.add_option("--no-highlights", dest='highlights', default=True, action='store_false',
                        help="disable highlights generation")
    parser.add_option("--no-grid", dest='grid', default=True, action='store_false',
                      help="disable grid on conf matrix")
    parser.add_option("--force-predicted-idxs", dest='force_predicted_idxs', default=False, 
                      action='store_true',
                      help="force predicted idxs to be actual idxs")
    parser.add_option("-d", dest='outputdir', default='.',
                        help="output directory, defaults to .")

    options, args = parser.parse_args()
    if options.fname1:
        tsv_file = options.fname1.endswith('.tsv')
        if tsv_file:
            if not options.fname2:
                sys.exit('Missing -2 option')
            ConfMatrix.gen_confusion_file(options.fname1, options.fname2, options.class_field, options.outputdir)
            print(('Loading labels from {} and {}'.format(options.fname1, options.fname2)))
            matrix = ConfMatrix.files2confmatrix(options.fname1, options.fname2, options.class_field, 
                                                 options.constraint, options.force_predicted_idxs)
        else:
            print(('Loading labels from {}'.format(options.fname1)))
            matrix = ConfMatrix.load_matrix(options.fname1)
    else:
        print('Missing -f option')
        sys.exit(1)

    output = options.output or options.fname1
    if tsv_file:
        output = "%svs%s.confusion" %(os.path.basename(options.fname1).replace('.tsv',''),
                                      os.path.basename(options.fname2).replace('.tsv',''))

    if options.fname2 and not options.fname2.endswith('.tsv'):
        print(('Loading labels from {}'.format(options.fname2)))
        matrix2 = Matrix.load_matrix(options.fname2)
        matrix.d-= matrix2.d
         
    matrix.gen_conf_matrix(output, matrix.labels, title=options.title,
                           threshold=options.threshold, fontsize=options.fontsize,
                           highlights=options.highlights, hthreshold=options.hthreshold,
                           grid=options.grid, output_dir=options.outputdir)
