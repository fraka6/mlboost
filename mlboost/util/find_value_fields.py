#!/usr/bin/env python
""" find fied of a given value in a tsv file """

import csv, sys
import argparse
parser = argparse.ArgumentParser(description=str(__doc__),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-v", dest="values", default=None,
                    help="comma separated values to find indexes")
    
args = parser.parse_args()
values = args.values.split(",")
    
reader = csv.DictReader(sys.stdin, delimiter="\t", restval='?', quoting=csv.QUOTE_NONE)

for i, row in enumerate(reader):
    
    fields = [field for field in reader.fieldnames if row[field] in values]
    print("line #%i (%s)" %(i, ",".join(fields)))



