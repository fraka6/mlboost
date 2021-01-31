''' create a simple interface to wornet synonyms '''
from nltk.corpus import wordnet as wn

from .synonyms import Synonyms
# cach entity synonyms for efficiency
_cache = {}

class WSyns(Synonyms):
    ''' synonyms cllass from wordnet (nltk)'''
    @staticmethod
    def get(entity, default=None):
        ret = WSyns.get_synonyms(entity)
        if default!=None and len(ret)==0:
            return default
        else:
            return ret
    
    @staticmethod
    def get_synonyms(entity):
        ''' get entity synonyms from wordnet'''
        if entity in _cache:
            return _cache[entity]
        synsets = wn.synsets(entity)
        synonyms = set()
        for s in synsets:
            synonyms.update(s.lemma_names)
        synonyms = synonyms
        _cache[entity]=synonyms
        return list(synonyms)

if __name__ == '__main__':
    import sys
    entity = sys.argv[1]
    print(WSyns.get_synonyms(entity))

