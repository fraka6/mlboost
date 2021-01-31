#!/usr/bin/env python
''' simple wrapper to search (cmdline) or server'''
''' example:
    - title:"The Right Way" AND text:go
    - "jakarta apache" jakarta (or default)
    - "jakarta apache" AND "Apache Lucene"
    - "jakarta apache" NOT "Apache Lucene"
    - (jakarta OR apache) AND website
    - proximity searches: "jakarta apache"~10" (For example to search for a "apache" and "jakarta" within 10 words of each other in a document use the search)
    - fuzzy match: roam~
    doc lucene search: http://lucene.apache.org/core/4_3_0/queryparser/org/apache/lucene/queryparser/classic/package-summary.html '''

INDEX_DIR = "IndexFiles.index"
MAX_N = 200
import sys, os, lucene
import logging

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import KeywordAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory, RAMDirectory, MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version

DEFAULT_SEARCH_FIELD = 'content'

"""
This script is loosely based on the Lucene (java implementation) demo class 
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it
will search the Lucene index in the current directory called 'index' for the
search query entered against the 'contents' field.  It will then display the
'path' and 'name' fields for each of the hits it finds in the index.  Note that
search.close() is currently commented out because it causes a stack overflow in
some cases.
"""

class Index:
    def __init__(self, searcher, analyzer, verbose=False):
        self.searcher = searcher
        self.analyzer = analyzer
        self.verbose = verbose    

    def get(self, query, field=DEFAULT_SEARCH_FIELD, n=MAX_N):
        ''' get that includes options parsing '''
        
        query.strip()
        entity = query
        if ":" in query:
            qsplit = query.split(':')
            entity, options = qsplit[0], qsplit[1:]
            
            for opt in options:        
                if opt.isdigit():
                    n = int(opt)
                else:
                    logging.warning('unknown opts %s' %opt)

        query = QueryParser(Version.LUCENE_CURRENT, field, analyzer).parse(entity)
        scoreDocs = searcher.search(query, n).scoreDocs
        
        results=[]
        if self.verbose:
            print("%s total matching documents for <%s>." %(len(scoreDocs), entity))
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            fname = doc.get("filename")
            #content = doc.get("content")
            #print "content size", len(content)
            results.append(fname)
            
        return results

    def prompt(self,  max_n=MAX_N):
    
        # by default we check content
        field = DEFAULT_SEARCH_FIELD
    
        while True:
            
            n = max_n
            
            print("Hit enter with no input to quit.")
            query = input("Query:").strip().lower()
        
            if query == '':
                return
        
            results = self.get(query, field,  n)
     
            if len(results)==0:
                print("0 results!")
        
            for i, el in enumerate(results):
                print("-----------------------")
                if isinstance(results, dict):
                    details = str(results[el]).encode('utf-8')
                    print('#%i entity: %s %s' %(i+1, el, details))
                else:
                    details = el.encode('utf-8')
                    print('#%i entity: %s' %(i+1, details))

if __name__ == '__main__':
    from optparse import OptionParser
    op = OptionParser(__doc__)
    
    op.add_option("-d", default=INDEX_DIR, dest="index_dir", 
                  help="index dir; default = %s" %INDEX_DIR)
    op.add_option("--all_content", default=False, action='store_true', 
                  dest="all_content", help="add file content not line")
    op.add_option("--stdanalyzer", default=False, action='store_true', 
                  dest="stdanalyzer", 
                  help="set analyzer to standardanalyzer no keyword analyzer")
    op.add_option("-s", default=False, action='store_true', 
                  dest="server", help="server mode")
    op.add_option("-p", default=8080, type='int', dest="port", help="port #")
    op.add_option("--std", default=False, dest="std", action='store_true', help="set to std search not exact match") 
    op.add_option("-V", default=False, dest="verbose", action='store_true', help="set verbose to true")
    op.add_option("--simplefs", default=False, dest="simple_fs", action='store_true', help="force SimpleFSDirectory instead if MmapDirectory")

    opts, args = op.parse_args(sys.argv)
        
    lucene.initVM()
    print('lucene', lucene.VERSION)
    
    if opts.stdanalyzer: 
        print("creating stdanalyzer -> keyword match on %s" %DEFAULT_SEARCH_FIELD)
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
    else:
        print("creating keyworkanalyzer -> exact match on %s" %DEFAULT_SEARCH_FIELD)
        analyzer = KeywordAnalyzer(Version.LUCENE_CURRENT)

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    fname = os.path.join(base_dir, opts.index_dir)
    print("loading index:", fname)
    
    if opts.simple_fs:
        directory = SimpleFSDirectory(File(fname))
    else:
        directory = MMapDirectory.open(File(fname))

    dir_reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(dir_reader)
    
    index = Index(searcher, analyzer, not opts.std, opts.best, 
                                 opts.verbose)

    if opts.server:
        from . import server
        server.run(opts.port, index)
    else:
        prompt(index, opts.max_n)
    
    del searcher
