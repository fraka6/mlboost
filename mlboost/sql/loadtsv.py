#!/usr/bin/env python
''' load tsv files to mysql 
    examples: python loadtsv.py --db test -c 201508/0*.tsv 
              python loadtsv.py --db test -c 2015 --year 
              python loadtsv.py --db year -t stage2 -c 2014 --year --mysql_opt="--sock=/work/mysql_stage2/mysql.sock"
'''
from datetime import date, timedelta
import os
path = "/nrg1/static09/vault/mobile/drops/nlu/ncs5/dma/stable/tsv/stage02/DRAGON_D2C_200_NMAID_20121105"#201509/08.tsv
yesterday = date.today()-timedelta(days=1)
default_constraint = "%i%02d/%02d.tsv" %(yesterday.year, yesterday.month, yesterday.day)

import argparse
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--db", default='prod', help="database name")
parser.add_argument("-t", dest="table", default='stage2', help="table name")
parser.add_argument("-c", dest="constraint", default=default_constraint, help="table name")
parser.add_argument("--year", action="store_true", help="load entire year (see example)")
parser.add_argument("--mysql_opt", help="mysql options to add (ex --sock='/work/mysql')")

args = parser.parse_args()


cmd = 'echo "use %s;" >load.sql' %args.db
os.system(cmd)
if args.year:
    for i in range(1,13):
        cmd = "ls %s/%s%02d/*.tsv|sed \"s|\(.*\)|LOAD DATA LOCAL INFILE '\\1'  INTO TABLE %s  FIELDS TERMINATED BY '\\t'  LINES TERMINATED BY '\\n' IGNORE 1 LINES;|\">>load.sql" %(path, args.constraint, i, args.table)
        print(cmd)
        os.system(cmd)
else:
    cmd = "ls %s/%s | sed \"s|\(.*\)|LOAD DATA LOCAL INFILE '\\1'  INTO TABLE %s  FIELDS TERMINATED BY '\\t'  LINES TERMINATED BY '\\n' IGNORE 1 LINES;|\">>load.sql" %(path, args.constraint, args.table)
    print(cmd)
    os.system(cmd)

cmd="cat load.sql | mysql -u root %s" %args.mysql_opt
print(cmd)
os.system(cmd)

