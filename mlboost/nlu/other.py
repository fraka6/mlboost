''' generate adjectif synonyms dictionary from files 
bzcat data.gz |grep "Rac" -A 2 |grep -B1 -A1 adj | grep -v "\-\-" | python other.py  
or
cat other.in | python other.py

remark (warning):  grep -v "adj$" ...doesn't work (not sure why)

example of lines treated:

Rac #:6360 - great, fabulous
        adj
                admirable awesome banner2 beyond-compare blue-chip boffo2 capital6 dandy epic2 excellent exceptional exc. exquisite fabulous fantastic farout five-star four-star glorious golden good_aspect good_event good_adj grand great heavenly2 incredible magnificent majestic mind-blowing outstanding2 recherche2 resplendent spectacular splendid stupendous sublime super2 superb superior2 superlative swell3 terrific top-hole topflight unsurpassed wonderful

'''
import os
from util import Synonyms 
import re
import string
import pickle as pickle

_default_fname = 'adj_dict.p'
 
punctuation_pattern="[%s]" %re.escape(string.punctuation)

def rm_ponctuation(entities):
    if not isinstance(entities, list):
        entities = [entities]
    return [re.sub(punctuation_pattern, ' ', entity) for entity in entities]

class CSyns(Synonyms):
    ''' synonyms dictionary from other '''
    def __init__(self, fname=None, verbose=False):
        self.verbose=verbose
        if fname:
            self.load(fname)

    def save(self, fname):
        print("saving %s (size=%i)" %(fname, len(self)))
        f = open(fname, 'w')
        pickle.dump(dict(self), f)
        f.close()
        
    def load(self, fname):
        print("loading...")
        dirname = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(dirname, fname), 'r')
        self.update(pickle.load(f))
        print("%s (size=%i)" %(fname, len(self)))
        f.close()

    def add(self, entity, synonyms):
        entity = re.sub(punctuation_pattern, ' ', entity)
        synonyms = [re.sub(punctuation_pattern, ' ', el) for el in synonyms]
        if entity in self:
            self[entity].update(synonyms)
            if self.verbose:
                print("updating %s -> %s" %(entity, self[entity]))
        else:
            self[entity]=set(synonyms)
            if self.verbose:
                print("creating %s ->%s"  %(entity, self[entity]))
    
    def treat_next_entity(self):
        line=sys.stdin.readline().strip()
        if "Rac #:" not in line:
            return 
        entities = line.strip().split('-')[1][1:].split(',')
        if self.verbose:
            print("-----------------")
            print("entities: %s" %entities)
        if 'adj' not in sys.stdin.readline():
            return 
        syns = sys.stdin.readline().strip().split(' ')
        if self.verbose:
            print("synonyms (before cleanup):%s" %(syns))
        syns = [s for s in syns if not s[-1].isdigit()]
        if self.verbose:
            print("synonyms (after cleanup):%s" %(syns))
        for entity in entities:
            print("adding ", entity)
            self.add(entity, syns)
        if self.verbose:
            print("done")
            

if __name__ == '__main__':
    import sys
    from optparse import OptionParser
    op = OptionParser(__doc__)
    
    op.add_option("-f", dest='fname', default=_default_fname, help="pickle filename to dump adj dict")
    op.add_option("-v", dest='verbose', default=False, action='store_true', help="activate verbose")
    opts, args = op.parse_args(sys.argv)
    
    syns = CSyns(verbose=opts.verbose)
    try:
        # temporary hack to ensure raise exception at the EOF (while doesn't work ;(
        for i in range(100000):
            syns.treat_next_entity()
    except Exception as ex:
        print("Exception", ex)
        
    syns.save(opts.fname)
     
         
         
        
