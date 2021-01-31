#!/usr/bin/env python
''' generate top diff of 2 tsv files
    example: python tsvtopdiff.py file1.tsv file2.tsv
'''
import argparse
import csv
from collections import defaultdict
from mlboost.core.pphisto import SortHistogram
from pprint import pprint 
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("-1", dest="fname1", help="tsv filename 1")
parser.add_argument("-2", dest="fname2", help="tsv filename 2")
parser.add_argument("--top", default=20, type=int, help="top diff by field")
parser.add_argument("--min", default=5, type=float, help="top diff by field")
parser.add_argument("--fields", default=None, help="fields to check split by comma (default = all fields)")
parser.add_argument("--freq", default=[], help="fields to convert to freq count distribution (ex: ids)")
parser.add_argument("--perc", action="store_true", help="compare percentage change (apply to all files) default -> absolute difference")
args = parser.parse_args()


data1 = csv.DictReader(open(args.fname1, 'rb'), delimiter='\t')
data2 = csv.DictReader(open(args.fname2, 'rb'), delimiter='\t')

# check files
if data1.fieldnames!=data2.fieldnames:
    raise Exception("headers aren't compatible")

# check fields
if args.fields:
    fields = args.fields.split(',')
    not_included = [field for field in fields if field not in data1.fieldnames]
    if len(not_included)>0:
        raise Exception("missing fields %s" %not_included)        
# default fields (all)
else:
    fields = data1.fieldnames

if args.freq:
    args.freq = args.freq.split(',')

def ddict2dict(d):
    ''' defaultdict to dict converter'''
    for k, v in list(d.items()):
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

def get_field_dist(dr):
    ''' generate field dist of a dictreader '''
    dist={}; n=0
    # generate field freq dist container
    for field in fields:
        dist[field]=defaultdict(lambda: 0.0)
    # update counts for each fields
    for line in dr:
        n+=1
        for field in fields:
            dist[field][line[field]]+=1

    # normalise
    for field in fields:
        # treat exception (freq conversion)
        if field in args.freq:
            new_dist = defaultdict(lambda: 0.0)
            for k,v in list(dist[field].items()):
                new_dist[v]+=1
            dist[field]=new_dist

        for k in dist[field]:
            dist[field][k]=dist[field][k]/n*100
    return dist

def get_top_diff(dist1, dist2, n):
    ''' return the top n field difference '''
    diff={}
    # generate diff field freq dist container
    for field in list(dist1.keys()):
        diff[field]=defaultdict(lambda: 0.0)
    # compute diff
    for field in list(dist1.keys()):
        field_diff=diff[field]
        field_dist1=dist1[field]
        field_dist2=dist2[field]
        values = set(list(field_dist1.keys()) + list(field_dist2.keys()))

        # get absolute difference
        for val in values:
                field_diff[val]=abs(field_dist2[val]-field_dist1[val])
        # get percentage difference
        if args.perc:
            for val in values:
                # ref = dist 1; perc diff = abs(dist2-dist1)/dist1
                if field_dist1[val]==0:
                    field_diff[val]=100.0
                else:
                    field_diff[val]/=field_dist1[val]
            
        sh = SortHistogram(ddict2dict(field_diff), False, True)
        el = [el for el in sh[:n] if el[1]>=args.min]
        if len(el)>0:
            print("-------------------")
            print(field)
            pprint(el)
            
    return diff
            

dist1 = get_field_dist(data1)
dist2 = get_field_dist(data2)
get_top_diff(dist1, dist2, args.top)

