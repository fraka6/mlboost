#!/usr/bin/env python
''' cut.py is a smarther cut and consider better splitting '''
import sys
import csv

from season import get_season

if __name__ == '__main__':
    from optparse import OptionParser
    op = OptionParser(__doc__)
    op.add_option("-d", default=',', type="string", dest="delimiter", 
                  help="field delimiter")
    op.add_option("-f", '--fields', default=None, dest="fields", 
                  help="fields to select; coma seperater (can be idx or fieldname")
    op.add_option("-i", "--idx", default=False, action='store_true', 
                  dest="index", help="show field indexs")
    op.add_option("-s","--season", default=[], 
                  dest="season", help="field to convert to season; comma separated")
    op.add_option("-v","--verbose", default=False, action="store_true", help="activate verbose output") 
    
    opts, args = op.parse_args(sys.argv)

    reader = csv.DictReader(sys.stdin, delimiter=opts.delimiter)

    if opts.season!=[]:
        opts.season = opts.season.split(opts.delimiter)
    
    # create optimal fieldnames (replace idx by fieldnames)
    #print(reader.fieldnames)
    fieldidx = dict([(i, name) for i,name in enumerate(reader.fieldnames)])
    #print(fieldidx)
    if opts.fields==None:
        fields = reader.fieldnames
    else:
        fields = [el if (not el.isdigit() or el in fieldidx) else fieldidx[int(el)] for el in opts.fields.split(',')]
        #print("FIELDS:", fields)
    if opts.index:
        for idx, fieldname in enumerate(fields):
            print("{field} #{i}".format(field=fieldname, i=idx)) 
        sys.exit()
    
    def get_value(value, field):
        if field in opts.season:
            return get_season(row.get(field), opts.verbose)
        else:
            return value
        
    for row in reader:
        try: 
            #print(opts.delimiter.join([str(datetime.strptime(row.get(el),'%d/%m/%Y')) for el in fields]))
            if opts.verbose:
                print([row.get(field, "?") for field in fields])
                print(row)
            print(opts.delimiter.join([get_value(row.get(field, "?"), field) for field in fields]))
        except Exception as exception:
            print("error", exception, row)