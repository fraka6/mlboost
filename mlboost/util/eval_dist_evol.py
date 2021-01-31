#!/usr/bin/env python
""" simplify visualisation of a distribution list
usage example:
"""
import json
import sys
from io import StringIO
test_line = '[{"unknown": 6.0, "initiated": 3.0}, {"notFound_contact": 4.0, "unknown": 2.0, "accepted": 1.0, "initiated": 3.0}, {"notFound_contact": 1.0, "unknown": 6.0, "initiated": 2.0, "reset": 1.0}, {"reset": 1.0, "unknown": 4.0, "notSupported": 1.0, "initiated": 4.0}, {"cancelled": 1.0, "unknown": 4.0, "notSupported": 1.0, "initiated": 4.0}, {"cancelled": 2.0, "unknown": 4.0, "accepted": 1.0, "initiated": 3.0}, {"unknown": 6.0, "initiated": 4.0}, {"reset": 1.0, "unknown": 3.0, "notSupported": 2.0, "initiated": 4.0}, {"cancelled": 1.0, "unknown": 4.0, "notSupported": 4.0, "reset": 1.0}, {"unknown": 5.0, "initiated": 5.0}, {"unknown": 6.0, "initiated": 4.0}, {"unknown": 8.0, "notSupported": 1.0, "initiated": 1.0}, {"unknown": 6.0, "notSupported": 1.0, "initiated": 3.0}, {"unknown": 5.0, "initiated": 5.0}, {"cancelled": 1.0, "unknown": 4.0, "notSupported": 1.0, "initiated": 4.0}, {"cancelled": 1.0, "unknown": 3.0, "accepted": 1.0, "initiated": 5.0}, {"unknown": 3.0, "notSupported": 1.0, "initiated": 6.0}, {"initiated": 1.0}]'

def normalize(dist):
    total=sum([float(el) for el in list(dist.values())])
    for key in dist:
        dist[key]=float(dist[key])/total*100

def diff(dist1, dist2):
    keys=set()
    keys.update(list(dist1.keys()))
    keys.update(list(dist2.keys()))
    return sum([abs(dist1.get(k, 0)-dist2.get(k, 0)) for k in keys])

def process(line, args):
    ''' args: norm, output, non_zero, tree '''
    dists_str = line.strip()[1:-1].split('},')
    dists=[]
    keys=set()
    for dist in dists_str:
        dist = dist if dist[-1] == "}" else dist+"}"
        dist = eval(dist)
        if args.norm:
            normalize(dist)
        dists.append(dist)
        keys.update(list(dist.keys()))
    
    n=len(dists)
    if args.output=='print':
        for i, dist in enumerate(dists):
            if args.tree and not ( (i<args.tree) or (i>n-args.tree-1) or (i>n/2-args.tree and i<n/2+args.tree)):
                continue
            for key in keys:
                val = dist.get(key, 0)
                if (not args.non_zero) or (args.non_zero and val):
                    print(key, val)
            print("---")
    elif args.output=='diff':
        print([diff(dists[i], dists[i+1]) for i in range(len(dists)-1)])
            
                        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=str(__doc__),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  
    parser.add_argument("--dn", dest="norm", default=True, 
                        action="store_false", help="don't normalize")
    parser.add_argument("--test", dest="test", default=False, 
                        action="store_true", help="do a simple test")
    parser.add_argument("--diff", dest="diff", default=False, 
                        action="store_true", help="get dist diff")
    parser.add_argument("--nz", dest="non_zero", default=False, 
                        action="store_true", help="print only non zero")
    parser.add_argument("-o", dest="output", default='print', 
                        help="output format: print, diff, ...")
    parser.add_argument("-3", dest="tree", default=None, type=int,
                        help="check begining/middle/end (output = print)")
    args = parser.parse_args()

    if args.test:
        process(test_line, args)
        sys.exit(0)
    sys.stdin.readline()
    for line in sys.stdin.readlines():
        process(line, args)
        if args.output=='print':
            print("--------------------------------------")
    
