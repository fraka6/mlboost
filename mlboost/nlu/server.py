#!/usr/bin/env python
''' simple server http://localhost:8080/name_entity
example: echo "Francis" | server.py '''
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import json
import time
import logging
from mlboost.core.rtstats import rtstats

getter = {}
FORMATS = ('raw','html','json')
format = FORMATS[0]
verbose = False
wind_stat = 100

class EntityHandler(BaseHTTPRequestHandler):
        
    '''def __init__(self, request, client_address, server):
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)
        self.n_requests = 0
        self.format = format
        self.stats = rtstats()
       ''' 
    #handle GET command
    def do_GET(self): 
        
        self.n_requests+=1
        start_time = time.time()

        ne = self.path[1:].replace('_', ' ').lower().strip()
        ne = ne.replace('%20',' ')
        if verbose:
            print("checking \"%s\"" %ne)
        if ne=="ALL":
            self.request.sendall("%s" %(list(NE_dict.keys())))
        elif format == 'html':
            self.send_response(200)
            self.send_header("Content-type", "text/plain") 
            self.send_header('Content-type','text-html')
            self.end_headers()
            self.wfile.write("%s\t%s" %(ne, NE_dict.get(ne)))
        elif format == 'json':
            self.request.sendall(json.dumps({ne:NE_dict.get(ne)}))
        else:
            self.request.sendall("%s\t%s" %(ne, NE_dict.get(ne)))
        
        self.stats.append(time.time()-start_time)
        if self.n_requests%wind_stat == 0:
            logging.info("mean=%.sf sec std=%.2f n=%i" %(self.stats.means, self.stats.stddev,
                                                         self.n_requests))
            
        return
    
def run(port=8080, getter_obj=getter):
    global getter
    getter = getter_obj or getter
    print('http server is starting...')
    #ip and port of server
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, EntityHandler)
    print(('http server is running on port %i...' %port))
    httpd.serve_forever()
    
if __name__ == '__main__':
    from optparse import OptionParser
    op = OptionParser()
    op.add_option("-p", default=8080, type="int", dest="port", help="port #")
    op.add_option("-f", default=FORMATS[0], dest="format", help="format %s" %(str(FORMATS)))
    
    (opts, args) = op.parse_args(sys.argv)
    
    format = opts.format
    
    run(opts.port, None)
