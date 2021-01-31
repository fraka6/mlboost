#!/usr/bin/env python
''' generate create table from a adksummary (tsv) files
    example: python tsv2table.py --db test |mysql -u root
'''
from datetime import date, timedelta

path = "/nrg1/static09/vault/mobile/drops/nlu/ncs5/dma/stable/tsv/stage02/DRAGON_D2C_200_NMAID_20121105/"#201509/08.tsv
yesterday = date.today()-timedelta(days=1)
fname = "%s/%i%02d/%02d.tsv" %(path, yesterday.year, yesterday.month, yesterday.day)

import argparse
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("-f", dest="fname", default=fname, help="tsv filename")
parser.add_argument("-t", dest="table", default='stage2', help="table name")
parser.add_argument("--db", default='prod', help="database name")
parser.add_argument("--basic", action="store_true", help="add min index")
parser.add_argument("--myisam", action="store_true", help="add myisam constraint")
parser.add_argument('--pc', dest="primary_key", action="store_true", help="had primary key not auto increment")
args = parser.parse_args()

# load fields and data example
f = open(args.fname, 'r')
fields = f.readline().strip().split('\t')
data = f.readline().strip().split('\t')

# create the create table
sql = "use %s; CREATE TABLE %s(" %(args.db, args.table)
for field, data in zip(fields, data):
    sql+=field
    if field == 'tsTime':
        sql+=" DATETIME,"
    elif data.isdigit():
        sql+=" INT,"
    elif field == 'callOuts':
        sql += " TEXT,"
    else:
        sql+=" VARCHAR(255),"

if args.primary_key:
    sql+="PRIMARY KEY (session, tsStep))";
else:
    sql = sql+"id INT PRIMARY KEY AUTO_INCREMENT)"

if args.myisam:
        sql+=";";
else:
    sql+="TYPE=MYISAM;"

if args.basic:
    for key in ("series_finalStatus","series_domainNLU",'tsStep','tsTime'):
        sql+="ALTER TABLE `%s` ADD INDEX `%s` (`%s`);" %(args.table, key, key)
else:
    for key in ("series_finalStatus","series_domain","series_domainNLU",'tsStep','tsTime','tsCount','adk_version','nlps_version','nlu_version','clientVer','clientBuild','carrier','phoneType','micType','estimatedGender','drivingMode',"domainName","parseType","nlps_visualResponsePrompt","adk_fieldID","adk_fieldID_response", "nlps_responsePrompt_keys","series_dialogClass","series_edits","series_duration","codec","codecCode"):
        sql+="ALTER TABLE `%s` ADD INDEX `%s` (`%s`);" %(args.table, key, key)
        
        
print(sql)
