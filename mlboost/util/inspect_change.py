#!/usr/bin/env python
''' find response change version 
    example: python inspect_change.py -f file.tsv --field ans --var nlps_responsePrompt_keys --version clientVer -c "mentionList:loc,location:<" -i core_weather_location_unknown
'''

import argparse
import csv
from collections import defaultdict
import gzip
import io
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("-f", dest='fname', help="filename")
parser.add_argument("--field", help="field to search")
parser.add_argument("--var", help="field to check variation")
parser.add_argument("--version", help="version field to display")
parser.add_argument("--info", default='', help="fields info to display at the end (comma separated)")
parser.add_argument("--deli", help="split var on this delimiter and check first part (ex:a|b, check only a)")
parser.add_argument("-i", dest="include", default=None, help="one of the var field should include this string")
parser.add_argument("-c", dest="constraints", default="", 
                    help="constraint->format=key1:val1,key2,val2; basically the key contain this cal string")

args = parser.parse_args()

# get constraints list:
constraints=[el.split(':') for el in args.constraints.split(",")]
info_fields=[field for field in args.info.split(',')]

# open the right reader
if args.fname.endswith('.gz'):
    zfile = gzip.open(args.fname, 'rb')
    data = io.StringIO(zfile.read(csv))
    reader = csv.reader(data)
else:
    reader = csv.DictReader(open(args.fname, 'rb'), delimiter='\t')

# create version and var field container (key=args.field)
version = defaultdict(lambda:[])
var = defaultdict(lambda:[])
info = defaultdict(lambda:[])

# load version and var
for row in reader:
    skip=False
    for key,val in constraints:
        if val not in row[key]:
            skip=True
    if not skip:        
        version[row[args.field]].append(row[args.version])
        var[row[args.field]].append(row[args.var])
        info[row[args.field]].append([row[field] for field in info_fields])
        

# get sequence change count 
change_seq=defaultdict(lambda:0)
change_ans=defaultdict(lambda:[])
change_info=defaultdict(lambda:[])

# create intermediate data structure
for ans, var_values in list(var.items()):
    version_values = version[ans]
    info_values = info[ans]
    # need at leat 2 values
    if len(var_values)<2:
        continue
    # need to include at least one value=include
    elif args.include!=None:
        if len([var for val in var_values if val.startswith(args.include)])==0:
            continue
                
    for i in range(len(var_values)-1):
        varseq = var_values[i:i+2]
        verseq = version_values[i:i+2]
        infoseq = info_values[i:i+2]
        deli = args.deli
        if ((deli and (varseq[0].split(deli)[0] not in varseq[1].split(deli)[0])) or
        (not deli and (varseq[0] not in varseq[1]))):
            key = '->'.join(varseq)+':'+'->'.join(verseq)
            change_seq[key]+=1
            change_ans[key].append(ans)
            change_info[key].append(infoseq)
                
# get change info
for var, count in list(change_seq.items()):
    print("%s -> %s times for:%s; info <%s>" %(var, count, change_ans[var], change_info[var]))
    
        
