class Data:pass

from os import path
import sys
from mlboost.util.file import open_anything

def load(fname, ratio=.7, sep='\t', target=None):
    ''' loads default format 
        format: default = features, target 
        you can choose the class column name (default is last idx)'''
    data = []
    target_vals = []
    if not path.isfile(fname):
        sys.exit("ERROR: file %s doesn't exit" %fname)
    reader = open_anything(fname, 'r')
    print("loading {}".format(fname))
    features = reader.readline().strip().split(sep)
    if len(features)>1:
        target = features[-1] if not target else features.index(target)
    else:
        target = ['?']
    for line in reader:
        features = line.strip().split(sep)
        if len(features)>1:
            data.append(features[:-1])
            target_vals.append(features[-1])
        else:
            data.append(features[0])
            target_vals.append('?')
    
    train = Data()
    test = Data()
    
    idx = int(ratio*len(data))
    
    # set fieldnames
    train.features = features
    test.features = features
    # set data
    train.data = data[0:idx]
    test.data = data[idx:]
    # set targets
    train.target = target_vals[0:idx]
    test.target = target_vals[idx:]
    

    return train, test, target

    
    
