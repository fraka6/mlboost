''' simple interface to nltk or loaded dictionary synonyms dictionaries 
    + simple prompt to play with the dictionary

example usage: python synonymes (prompt to test nltk synonymes)
'''
from os import path
import pickle
import nltk
from . import wn 
_example_fname = 'adj_dict.p'

def get_stop_words():
    return nltk.corpus.stopwords.words('english')

class Synonyms(dict):
    ''' synonyms dictionary basic interface '''
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
        if path.isfile(fname):
            print("loading...")
            dirname = path.dirname(path.abspath(__file__))
            f = open(path.join(dirname, fname), 'r')
            self.update(pickle.load(f))
        print("%s (size=%i)" %(fname, len(self)))
        f.close() 

    def filterInText(self, docs, query):
        ''' filter docs with query ''' 
        adjs = [word for word in query.split(' ') and word in self]
        
        syns = []
        for adj in adjs:
            adj_list = [adj]
            adj_list.extend(self.get(adj, []))
            syns.append(adj_list)

        # filter docs
        filtered_docs = []  
        for doc in docs:
            in_text =[]            
            for words in syns:
                # if text contain the adj or one of its synonyms, keep the product
                if True in [(word in doc) for word in words]:
                    filtered_docs.append(docs)
                
        return filtered_docs

    def prompt(self, dictionary):
        ''' prompt inteface to the dictionary d '''
        while True:
            print("Hit enter with no input to quit.")
            query = input("Query:").strip()
        
            if query == '':
                sys.exit(0)
            elif "eval:" == query[:5]:
                print(eval(query[5:]))
            else:
                print(self.get(query,'?'))

if __name__ == '__main__':
    import sys
    from optparse import OptionParser
    op = OptionParser(__doc__)
    
    op.add_option("-f", dest='fname', default=None, help="pickle filename dictionary (ex: %s)" %_example_fname)
    op.add_option("-d", dest='download',default=False, action="store_true", help="call nltk.download()")

    opts, args = op.parse_args(sys.argv)
    
    if opts.download:
        nltk.download()
    
    if opts.fname:
        syns = Synonyms(opts.fname)    
    else:
        syns = wn.WSyns()
    
    syns.prompt(syns)
     
