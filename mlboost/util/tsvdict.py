#!/usr/bin/env python
""" add field or field ratio columns from a dictionary field column
    ex: cat file.tsv |./tsvdict.py field --key key --ratio --only
    ex: cat file.tsv |./tsvdict.py field --sum
"""
import json
import sys
import argparse
import csv, sys
TRANSFORMATIONS=["ratio","sum"] 

parser = argparse.ArgumentParser(description="compute dict ratio of a field"
                                 "tsv FILE.")
parser.add_argument("field", metavar="FIELD",
                    help="selected dict field to compute ratio")
parser.add_argument("--key", default=None, help="selected key of the dict")
parser.add_argument("--strin", default=False, action="store_true",
                    help="key is a substring that should be in the key")
parser.add_argument("--json", action='store_true', default=False , help='load field as a json object')
parser.add_argument("--only", action='store_true', default=False , help='extract field only')
parser.add_argument("--trans", dest='trans', default=None,
                    help="set transformation (%s)" %TRANSFORMATIONS)
parser.add_argument("--ratio", action='store_true', default=False , help='create field.key ratio -> same as --trans ratio')
parser.add_argument("--sum", action='store_true', default=False , help='create field sum -> same as --trans sum')
parser.add_argument("--no_header", dest='header', action='store_false', default=True , help='remove header')


args = parser.parse_args()

new_field = "{field}".format(field=args.field)

if args.key:
    new_field+= ".{key}".format(key=args.key)
    
if args.ratio:
    args.trans='ratio'
if args.sum:
    args.trans='sum'

if args.trans:
    new_field += ".{trans}".format(trans=args.trans)

reader = csv.DictReader(sys.stdin, delimiter="\t", restval='?', quoting=csv.QUOTE_NONE)
reader.fieldnames.append(new_field)

# print header
if args.header:
    if args.only:
        print(new_field)
    else:
        print("\t".join(reader.fieldnames))

# jump header
next(reader)
# fields rows
for row in reader:
    # get dictionary
    if args.json:
        d = json.loads(row[args.field])
    else:
        d=eval(row[args.field])
    # get value
    if args.strin:
        keys = [key for key in list(d.keys()) if args.key in key]
        value = sum([d.get(key) for key in keys])
    else:
        value = float(d.get(args.key, 0))
    # apply transform if required
    if  args.trans:
        total = sum([int(el) for el in list(d.values())])
        if args.trans=='ratio':
            value/=total
        else:
            value=total
        
    row[new_field] = str(value)
    if args.only:
        print(value)
    else:
        print("\t".join([row[key] for key in reader.fieldnames]))
