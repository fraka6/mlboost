#!/usr/bin/python
''' rtstats = real time stats '''
# inspired by mlboost (http://sourceforge.net/projects/mlboost/)
# all rights given by the author: Francis Pieraut - 28 dec 2009
 
import math
from mlboost.core.pphisto import SortHistogram

try:
    from numpy import array
    def sum_square(els):
        vec=array(x)
        return (vec*vec).sum()
except:
    def sum_square(els):
        [x*x for x in els]
        return sum(els)
    
class rtstats:
    ''' rtstats = real time stats '''
    def __init__(self, x=None):
        self.sum=0.0
        self.n=0
        self.sum_squared=0.0
        # number of negative values
        self.nneg=0
        if x!=None:
            self.append(x)
    
    @classmethod
    def create(cls, sum, n, sum_squared, nneg):
        new = rtstats()
        new.sum = sum
        new.n = n
        new.sum_squared = sum_squared
        new.nneg = nneg
        return new

    def get_params(self):
        return self.sum, self.n, self.sum_squared, self.nneg

    def mean(self):
        if self.n==0:
            return 0
        return float(self.sum)/self.n

    def stddev(self):
        if self.n<=1:
            return 0
        try:
            return math.sqrt((self.sum_squared - ((self.sum*self.sum)/self.n))/(self.n-1))
        except:
            #std.err.writelines("ERROR in :rtstats.stddev()\n")
            print(self.sum_squared,self.sum,self.n)
            print((self.sum_squared - ((self.sum*self.sum)/self.n))/(self.n-1))
            return 0

    def append(self, x):
        x = float(x)
        self.n+=1
        self.sum+=x
        self.sum_squared+=x*x
        if x<0:
           self.nneg+=1
           
    def extend(self,els):
        
        self.n+=len(els)
        self.sum+=float(sum(els))
        self.sum_squared+=sum_square(els)
        self.nneg+=(els<0).sum()

    def clone(self):
        cl=rtstats()
        cl.sum=self.sum
        cl.n=self.n
        cl.sum_squared=self.sum_squared
        cl.nneg=self.nneg
        return cl

    def get_size_stats(self, d, fname=None):
        ''' generate stats of a dictionary element size or list element size'''
        stats={}
        total = 0
        for el in d:
            if isinstance(d, dict):
                size = len(d[el])
            else:
                size = len(el)
            total+=size
            self.append(size)
            if size in stats:
                stats[size]+=1
            else:
                stats[size]=1

        # normalize
        for el in stats:
            stats[el]=(100.0*stats[el])/len(d)

        print("mean:%2.2f stddev:%2.2f (total =%i)" %(self.mean(),self.stddev(), total)) 
        sh = SortHistogram(stats, False, True)
        print("size distribution:")
        for k,v in sh:
            print("%i -> %2.2f%%" %(k, v))
        if fname:
            f = open(fname,'w')
            import pickle
            pickle.dump(stats, f)
            f.close()


