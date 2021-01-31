''' file access & transformation util functions:
    - html highlighting
    - ....
'''
from os import path
import gzip
import zipfile
import urllib.request, urllib.parse, urllib.error, urllib.request, urllib.error, urllib.parse
import json
import logging 

current_path = path.dirname(__file__)
highlight_string = '<SPAN style="BACKGROUND-COLOR: #ffff00">TEXT</SPAN>'
n_chars_added = len(highlight_string)-len('TEXT')


def open_anything(fname, format=None, sep='\t'):
    ''' open the right reader depending on the format 
        format = default (None), csv, dictreader'''
    if fname.endswith('.gz'):
        reader = gzip.open(fname, 'rb')
    elif fname.endswith('.zip'):
        reader = zipfile.ZipFile(fname, 'rb')
    else:
        reader = open(fname, 'rb')
    
    if format == 'csv':
        reader = csv.reader(reader, delimiter=sep)
    elif format == 'dictreader':
        reader = csv.DictReader(reader, delimiter=sep)
    return reader

def query_server(ip='localhost'):
    encoded_query = urllib.parse.quote(query)
    url = 'http://%i/search/?query=%s' %(ip, encoded_query)
    request = urllib.request.Request(url)
    result = urllib.request.urlopen(request)
    matches = json.loads(result.read())
    return matches

class Files:
    ''' interface to files to retreive content '''
    def __init__(self, repo='', repo_raw=None, server=None):
        ''' server need a search(query) interface '''
        self.repo = repo
        self.repo_raw = repo_raw
        if server and not hasattr(server, 'search'):
            sys.stderr.writelines('server need a search(query) interfance')
        self.server = server

    def get_content(self, filename, raw=False):
        ''' get file content, useually used to get review content '''
        repo = self.repo_raw if raw else self.repo
        relative_fname = path.join(current_path, repo, filename)
        logging.debug("reading", relative_fname)

        if not path.isfile(relative_fname):
            logging.warning("file doesn't exist %s" %relative_fname)
            return ''

        f = open(relative_fname, 'r')
        content = f.read()
        f.close()
        return content

    def highlight_sentences(self, fname, sentences, raw=False):
        ''' fname contente html highlighting of sentences ''' 
        content = self.get_content(fname, raw)

        for sentence in sentences:
            content = content.replace(sentence, highlight_string.replace("TEXT", sentence))
            
        return content

    def highlight_sections(self, fname, start_end_idx_list, raw=False):
        ''' content sections html highlighting; start_end_idx is a list of start and end indexes'''
        
        content = self.get_content(fname, raw)
        def relative_idx(idx):
            ''' get update idx based on previous highlights added '''
            return idx+(i*n_chars_added)
        
        for i, (start, end) in enumerate(start_end_idx_list):
            start, end = relative_idx(start), relative_idx(end)
            text = content[start:end]
            content = content.replace(text, highlight_string.replace("TEXT", text)) 
        return content

    def get_sentences(self, fname, start_end_idx_list, raw=False):
        ''' highight sections of the file content start_end_idx is a list of start and end indexes'''
        sentences = []
        content = self.get_content(fname, raw)

        for start, end in start_end_idx_list:
            sentences.append(content[start:end])

        return sentences

    def query(self, query, ip=None):

        files = []
        if ip:
            matches = query_server(ip)
        else:
            matches = server.search(query)

        for match in matches:
            fname = match['docId']
            files.append(fname, self.highlight_sentences_idxs(fname, match['sentences']))
        return files

            
if __name__ == '__main__':
    import sys
    if len(sys.argv)>1:
        fname = sys.argv[1]
    else:
        fname = __file__

    files = Files()
    print("testing higlight section of a text\n----------------")
    print(files.highlight_sections(fname, [(10,20), (80,100)])[:120])
    print("get content")
    content =  files.get_content(fname)
    print(content[:50])
    print("testing higlight sentences of a text\n----------------")
    print(files.highlight_sentences(fname, [content[10:20], content[80:100]])[:120])
    
