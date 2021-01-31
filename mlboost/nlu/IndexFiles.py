#!/usr/bin/env python
''' Create lucene index 
example: python IndexFiles.py REPO
 note: do this if you have out of memory issues: export _JAVA_OPTIONS=-Xmx8g '''
INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
from datetime import datetime

from java.io import File
#from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import KeywordAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

def uopen(filename, mode='r'):
    import gzip, bz2
    ''' util open; don't have to worry about compress format anymore''' s
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

"""
This class is loosely based on the Lucene (java implementation) demo class 
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""

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
        print 'commit index',
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print 'done'

    def indexDocs(self, root, writer, allcontent=False, verbose=False, n=1000):
        ''' indexOptions choices : DOCS_ONY, DOCS_AND_FREQS, DOCS_AND_FREQS_AND_POSITIONS 
        ref: http://lucene.apache.org/core/old_versioned_docs/versions/3_5_0/api/core/org/apache/lucene/index/FieldInfo.IndexOptions.html '''
        # for entity 
        t1 = FieldType()
        t1.setIndexed(True)
        t1.setStored(True)
        t1.setTokenized(True)
        t1.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        t11 = FieldType()
        t11.setIndexed(True)
        t11.setStored(True)
        t11.setTokenized(True)
        t11.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

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

                print "\n#%i adding %s " %(i+1, category) 
                try:
                    path = os.path.join(root, filename)
                    
                    file = uopen(path)
                    
                    if allcontent:
                        doc = Document()
                        doc.add(Field("filename", filename, t1))
                        doc.add(Field("category", category, t1))
                        doc.add(Field("path", root, t1))
                        
                        content = unicode(file.read(), 'iso-8859-1')
                        file.close()
                        if len(content) > 0:
                            doc.add(Field("content", content, t3))
                        else:
                            print "warning: no content in %s" % filename
                        writer.addDocument(doc)
                    # index lines not full content
                    else:
                        for i, line in enumerate(file):
                            doc = Document()
                            #print category, ne, source, params doc.add(Field("filename", filename, t1))
                            doc.add(Field("filename", filename, t1))
                            doc.add(Field("category", category, t2))
                            doc.add(Field("content",line, t1))
                            #doc.add(Field("source", ne, t11))
                            #doc.add(Field("source", unicode(source,'iso-8859-1'), t2))
                            #doc.add(Field("params", unicode(params,'iso-8859-1'), t2))
                            writer.addDocument(doc)
                            if verbose:
                                print "adding \"%s\"->\"%s\" (%s:%s)" %(clean_ne, ne, source, params)
                        
                            if (i%n)==0:
                                sys.stdout.write('.')
                                sys.stdout.flush()

                except Exception, e:
                    print "Failed in indexDocs:", e

if __name__ == '__main__':

    from optparse import OptionParser
    op = OptionParser(__doc__)

    op.add_option("-d", default=INDEX_DIR, dest="index_dir", 
                  help="index dir; default = %s" %INDEX_DIR)
    op.add_option("--all_lines", default=False, action='store_true', 
                  dest="all_line", help="add line not file")
    op.add_option("--exactmatch", default=False, action='store_true', dest="exactmatch", 
                  help="set analyzer to keyword analyzer not  standardanalyzer")q
    op.add_option('-n', dest='n', default=1000, 
                  help="print . every n lines treated")
    opts, args = op.parse_args(sys.argv)
    
    if len(sys.argv) < 2:
        print IndexFiles.__doc__
        sys.exit(1)
    lucene.initVM(lucene.CLASSPATH, maxheap='8g')
    print 'lucene', lucene.VERSION
    start = datetime.now()

    if opts.exactmatch:
        print "creating keyworkanalyzer -> exact match"
        analyzer = KeywordAnalyzer(Version.LUCENE_CURRENT)
    else:
        print "creating stdanalyzer -> keyword match"
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
            
    try:
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        fname = os.path.join(base_dir, opts.index_dir)
        print "creating index:", fname
        IndexFiles(sys.argv[1], fname,analyzer, not opts.all_line)
        end = datetime.now()
        print end - start
    except Exception, e:
        print "Failed: ", e
        raise e
