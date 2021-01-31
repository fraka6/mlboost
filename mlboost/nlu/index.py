#!/usr/bin/env python
''' simple wrapper to create and use a lucene index '''

INDEX_DIR = "IndexFiles.index"

MAX_N = 200
import sys, os, lucene, threading, time
import logging

#general
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import KeywordAnalyzer
# Search
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory, RAMDirectory, MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version
# Index
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

import urllib.request, urllib.parse, urllib.error
import re
import logging

def uopen(filename, mode='r'):
    import gzip, bz2
    ''' util open; don't have to worry about compress format anymore'''
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    elif filename.endswith('.bz2'):
        return bz2.BZ2File(filename)
    else:
        if os.path.isfile(filename):
            return open(filename, mode)
        elif os.path.isfile(filename+'.gz'):
            return uopen(filename+'.gz')
        elif os.path.isfile(filename+'.bz2'):
            return uopen(filename+'.bz2')


DEFAULT_SEARCH_FIELD = 'content'

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, root, storeDir, analyzer, allcontent=False, n=1000):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        store = SimpleFSDirectory(File(storeDir))
        #analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(Version.LUCENE_CURRENT, analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        self.indexDocs(root, writer, allcontent, n=n)
        ticker = Ticker()
        print('commit index', end=' ')
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print('done')

    def indexDocs(self, root, writer, allcontent=False, verbose=False, n=1000):
        ''' indexOptions choices : DOCS_ONY, DOCS_AND_FREQS, DOCS_AND_FREQS_AND_POSITIONS 
        ref: http://lucene.apache.org/core/old_versioned_docs/versions/3_5_0/api/core/org/apache/lucene/index/FieldInfo.IndexOptions.html '''
        # for entity 
        t1 = FieldType()
        t1.setIndexed(True)
        t1.setStored(True)
        t1.setTokenized(True)
        t1.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        # for category & source
        t2 = FieldType()
        t2.setIndexed(False)
        t2.setStored(True)
        t2.setTokenized(False)
        t2.setIndexOptions(FieldInfo.IndexOptions.DOCS_ONLY)        

        # for content
        t3 = FieldType()    
        t3.setIndexed(True)
        t3.setStored(True)
        t3.setTokenized(True)
        t3.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        #t3.setIndexOptions(FieldInfo.IndexOptions.DOCS_ONLY) 
        
        
        for root, dirnames, filenames in os.walk(root):
            for i,filename in enumerate(filenames):
                category = filename.replace('.tsv','').replace('.bz2','')

                print("#%i adding %s " %(i+1, category)) 
                try:
                    path = os.path.join(root, filename)
                    
                    file = uopen(path)
                    
                    if allcontent:
                        doc = Document()
                        doc.add(Field("filename", filename, t1))
                        doc.add(Field("category", category, t1))
                        doc.add(Field("path", root, t1))
                        
                        content = str(file.read(), 'iso-8859-1')
                        file.close()
                        if len(content) > 0:
                            doc.add(Field("content", content, t3))
                        else:
                            print("warning: no content in %s" % filename)
                        writer.addDocument(doc)
                    # index lines not full content
                    else:
                        for i, line in enumerate(file):
                            doc = Document()
                            #print category, ne, source, params doc.add(Field("filename", filename, t1))
                            doc.add(Field("filename", filename, t1))
                            doc.add(Field("category", category, t2))
                            doc.add(Field("content",line, t1))
                            writer.addDocument(doc)
                            if verbose:
                                print("adding \"%s\"->\"%s\" (%s:%s)" %(clean_ne, ne, source, params))
                        
                            if (i%n)==0:
                                sys.stdout.write('.')
                                sys.stdout.flush()

                except Exception as e:
                    print("Failed in indexDocs:", e)


class Index:
    ''' lucene index search '''
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

    def prompt(self,  max_n=MAX_N, details=False):
    
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
            else:
                if not details:
                    print(results)
                else:
                    for i, el in enumerate(results):
                        print("-----------------------")
                        if isinstance(results, dict):
                            details = str(results[el]).encode('utf-8')
                            print('#%i entity: %s %s' %(i+1, el, details))
                        else:
                            details = el.encode('utf-8')
                            print('#%i entity: %s' %(i+1, details))

if __name__ == '__main__':
    from datetime import datetime
    from optparse import OptionParser
    op = OptionParser(__doc__)
    
    op.add_option("-d", default=INDEX_DIR, dest="index_dir", 
                  help="index dir; default = %s" %INDEX_DIR)
    op.add_option("--all_content", default=False, action='store_true', 
                  dest="all_content", help="add file content not line")
    op.add_option("--exact", default=False, action='store_true', 
                  dest="exact_match", help="set analyzer to standardanalyzer no keyword analyzer")
    op.add_option("-s", default=False, action='store_true', 
                  dest="server", help="server mode")
    op.add_option("-p", default=8080, type='int', dest="port", help="port #")
    op.add_option("-V", default=False, dest="verbose", action='store_true', help="set verbose to true")
    op.add_option("--simplefs", default=False, dest="simple_fs", action='store_true', help="force SimpleFSDirectory instead if MmapDirectory")
    op.add_option("--all_lines", default=False, action='store_true', 
                  dest="all_line", help="add line not file")
    op.add_option("-i", dest='create_index', default=False, action='store_true', help="create index; not search")
    op.add_option("--maxheap", dest='maxheap', default='8g', help="min ram for the VM")
    op.add_option("--max_n", dest='max_n', default=MAX_N, help="max return search item")
    opts, args = op.parse_args(sys.argv)
    
    lucene.initVM(maxheap=opts.maxheap)
    print('lucene', lucene.VERSION)
    start = datetime.now()

    if opts.exact_match: 
        print("creating keyworkanalyzer -> exact match on %s" %DEFAULT_SEARCH_FIELD)
        analyzer = KeywordAnalyzer(Version.LUCENE_CURRENT)
    else:
        print("creating stdanalyzer -> keyword match on %s" %DEFAULT_SEARCH_FIELD)
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    fname = os.path.join(base_dir, opts.index_dir)
    
    if opts.create_index:
        if len(sys.argv)<2:
            sys.stderr.writelines("ERROR: need a directory to index\n")
            sys.exit(1)
        try:
            print("creating index:", fname)
            IndexFiles(sys.argv[1], fname, analyzer, not opts.all_line)
            end = datetime.now()
            print(end - start)
        except Exception as e:
            print("Failed: ", e)
            raise e
        print("loading index:", fname)
        
    else:
        print("creating index...")
        if opts.simple_fs:
            directory = SimpleFSDirectory(File(fname))
        else:
            directory = MMapDirectory.open(File(fname))
            
        dir_reader = DirectoryReader.open(directory)
        searcher = IndexSearcher(dir_reader)
        
        index = Index(searcher, analyzer, opts.verbose)

        if opts.server:
            from . import server
            server.run(opts.port, index)
        else:
            index.prompt(opts.max_n)
    
        del searcher
       
