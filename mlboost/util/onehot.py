#!/usr/bin/env python
''' one hot discret field converter
    1 -> to idx dynamically
    2 -> idx to one hot vector
     '''
#from  collections import OrderedDict
from os import path
import itertools
import logging
from numpy import zeros, array
import pickle

class OneHot(object):
    def __init__(self, fieldnames, discret_fields):
        self.fieldnames = fieldnames
        self.discret_fields = discret_fields
        self.choices={field: [] for field in discret_fields}
        self.fallback_choices = {}
        self.default_fname = 'choices.p'

    def save(self, fname=None):
        fname = fname or self.default_fname
        with open(fname, 'wb') as f:
            pickle.dump(self.choices, f)
    
    def info(self):
        print("----------------------------------------\none hot discret expansion info:")
        for field in self.discret_fields:
            print("%s -> %i" %(field, len(self.choices[field])))

    def load(self, fname=None):
        fname = fname or self.default_fname
        if path.isfile(fname):
            with open(fname, 'rb') as f:
                self.choices = pickle.load(f)
        else:
            logging.error("%s doesn't exist")
        
    def val2idx(self, key, value):
        if value in self.choices[key]:
            return list(self.choices[key]).index(value)
        elif key in self.fallback_choices:
            logging.warning('Value %s not in choices for key %s; using fallback', value, key)
            return self.fallback_choices[key]
        else:
            self.choices[key].append(value)
            return len(self.choices[key])-1

    def get_vecfieldnames(self):
        new_header = []
        for field in self.fieldnames:
            if field in self.discret_fields:
                new_header.extend("%s_%s" %(field, v) for v in self.choices[field])
            else:
                new_header.append(field)
        return new_header

    def idx2vec(self, key, idx, line_i=None):
        size = len(self.choices[key])
        vec = [0]*size #zeros(len(self.choices[key]))
        if isinstance(idx, int) or (isinstance(idx,str) and idx.isdigit()):
            idx=int(idx)
            if idx>size:
                logging.warning("idx (%i)>size(%i)" %(idx, len(vec)))
            else:
                vec[idx]=1
        else:
            line_msg="" if not idx else " at line %i" %line_i
            logging.warning("line #%i not an idx=<%s> (field=<%s>;choices=%s)" %(line_i, idx, key, str(self.choices[key])))
                
        return vec        

    def row2idx(self, row, dictrow=True):
        "return a dictionary or a row"
        def field_value(field):
            return row[field] if field not in self.discret_fields else self.val2idx(field, row[field])
        if dictrow:
            return {field: field_value(field) for field in self.fieldnames}
        else:
            return [field_value(field) for field in self.fieldnames]

    def row2vec(self, row, i=None):
        "convert dictrow (with idx) to vector row; return a list"
        new_row = [[row[field]] if field not in self.discret_fields else self.idx2vec(field, row[field], i) for field in self.fieldnames]
        
        return list(itertools.chain.from_iterable(new_row))

if __name__ == "__main__":
    # create example : name, age
    data =[{"name":"Antoine",'age':8,'sex':'male'}, {"name":"Laurence",'age':4,'sex':'female'},{"name":"Francis",'age':38,'sex':'male'}, {"name":"Caro",'age':39,'sex':'female'}]
    
    oh = OneHot(['name', 'age','sex'], ['name','sex'])
    print("idx tranform...")
    data1=[]
    for row in data:
        new_row = oh.row2idx(row)
        print(row,"->",new_row)
        data1.append(new_row)

    oh.info()

    print("idx to vec transform....")
    for row in data1:
        print(row,"->",oh.row2vec(row))
    
    # try onehot sklearn encoding (need to load everything in memory)
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()

    print("now trying sklearn.OneHotEncoder....")
    new_data = []
    for row in data:
        new_data.append(oh.row2idx(row, False))
    # fit data and transform
    enc.fit(new_data)
    trans_data=enc.transform(new_data).toarray()
    for i,row in enumerate(new_data):
     print("%s->%s" %(row,trans_data[i]))
