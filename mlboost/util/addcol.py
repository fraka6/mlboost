#!/usr/bin/env python
''' simple script to add columns based on rules like 
    - C=A+B -> create a new column C and assign A+B
    - C=A>B -> create a column C and assign bool value 

example: 
          cat file.tsv | python addcol.py -a C=(A-B)/C
          cat file.tsv | python addcol.py -e C=A>5
          cat file.tsv | python addcol.py -e C=A:b>5    (row['A']['b']>5)
          cat file.tsv | python addcol.py -e C=5+ in (A)    (e in row['A'])
'''


import csv
import sys
import re
import json
import logging
EXAMPLE = "C=(A-B)/C" 

if __name__ == '__main__':
    from optparse import OptionParser
    op = OptionParser(__doc__)

    op.add_option("-a", dest='add', default="", 
                  help="add field(s) from this rules")
    op.add_option("-b", dest='addbool', default="", 
                  help="set bool field(s) from this rules")
    op.add_option("-v", dest='verbose', default=False, action="store_true", 
                  help="debug verbose")
    op.add_option("-c", dest='cast', default='float', 
                  help="cast value fct; default [[%DEFAULT%]]; nothing set=''")
    op.add_option('-d', dest='delimiter',default=',', help="set delimiter")
    opts, args = op.parse_args(sys.argv)
    
    def get(key='%s', row=True, cast=''):
        '''define value fct; ex return float(row['%s'])
           key format = %s, key or keyA:keyB
           return choices: %s, row['%s'], cast(X), json.loads(row['A'])['B'] 
        '''
        rowaccess="row['%s']"
        if ':' in key:
            keyA,key=key.split(':')
            rowaccess="json.loads(row['{keyA}']).get('%s','0')".format(keyA=keyA) 
    
        input = rowaccess %key if row else '%s'%key
        return input if cast=='' else cast+"(%s)"%input
    
    def clean(els):
        return [el for el in els if el!='']

    # basic option check
    if opts.add==None or opts.addbool==None:
        logging.errors("Nothing to do buddy")
    
    reader = csv.DictReader(sys.stdin, delimiter=opts.delimiter)
    header = reader.fieldnames
    
    # create evaluable rule
    rules = []
    add_rules = clean(opts.add.split(',')) 
    add_bool_rules = clean(opts.addbool.split(','))
    all_rules = list(add_rules)+list(add_bool_rules)
    
    logging.info("adding field")
    for i, rule in enumerate(all_rules):
        logging.info("%i -> %s" %(i+1, rule))
    
    logging.info("generating add colomn code...")
    for rule in add_rules:
        new_rule = str(rule)
        new_field = rule.split('=')[0]
        header.append(new_field)
        variables = list(set([key for key in re.findall("[^+-/*=()]*", rule) if key!='']))
        variables.sort(key=len, reverse=True)
        for key in variables:
            if key==new_field:
                new_rule = new_rule.replace(key, get(key))
            else:
                new_rule = new_rule.replace(key, get(key, cast=opts.cast))

        rules.append(new_rule)

    logging.info("generating add bool colomn code...")    
    # treat C=A>5
    operator='[=+-/><]*'
    w="[a-zA-Z0-9:_]*"

    for rule in add_bool_rules:
        if " in (" in rule:
            m=re.search("({w})=({w})[+]{0} in ({w})".format(w=w), rule)
            keyvals={'newkey':get(vals[0]),'value':get(vals[1],False), 
                     'key': get(vals[3],False)}
            if "+" in rule:
                keyval['value']=get(vals[1],False,'float')
                fctstr="{newkey}='1' if {value} >= max([float(v) for v in {key}])".format(**keyvals)
            else:
                fctstr="{newkey}='1' if {'value'} in {key}".format(**keyvals)
        else:
            m=re.search("({w})=({w})({op})({w})".format(op=operator, w=w), rule)
            header.append(m.group(1))
            vals=m.groups()
            keyvals={'newkey':get(vals[0]),'key':get(vals[1],cast=opts.cast), 
                     'operator':get(vals[2], False), 'value': get(vals[3],False, cast=opts.cast)}
            fctstr = "{newkey}='1' if {key} {operator} {value} else '0'".format(**keyvals)
        rules.append(fctstr)
        
    logging.info("generating new data")
    print(opts.delimiter.join(header))
    for row in reader:
        for rule in rules:
            exec(rule)
            if opts.verbose:
                key = rule[:rule.index(']')+1]
                print(key,"->",eval('"%s"' %key), "<%s>" %rule)
        print(opts.delimiter.join([str(row[el]) for el in header]))
        
             
