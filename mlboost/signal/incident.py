''' domain incident detection '''
import pylab
import pandas as pd
import numpy as np
from collections import defaultdict
from mlboost.core.pphisto import SortHistogram
pd.set_option('max_colwidth', 40)
pd.set_option("display.colheader_justify", "left")

def get_incidents(series, factor=2, quantile=None, 
                  min_ratio=.01, details=False, 
                  verbose=False, show=False):
    
    dates = pd.Series()
    total = series.sum().sum()
    # keep variation vak[date][key]=value/threshold
    ratiovar = defaultdict(lambda:defaultdict(lambda:0))
    ratios = dict()

    for i, (key, v)  in enumerate(series.items()):
        data=series[key]

        ratio = float(data.sum())/total
        ratios[key]=ratio
        print("analysing %s %2.2f%%" %(key, ratio*100))
        diff=data.diff().abs()
        
        # remove day trend
        diff2=(diff-pd.rolling_median(diff,3)).abs()
        
        # compute stats
        mean = diff2.mean()
        std = diff2.std()
        
        if quantile!=None:
            threshold=diff2.quantile([quantile])[quantile]*factor
        else:
            threshold=(mean+3*std)*factor
        
        # get incidents dates
        incidents = diff2[diff2>threshold]

        #if ratio >min_ratio:
        for date, value in incidents.items():
            total_date = series.ix[date].sum()
            val = dates.get(date, {})
            ratio = float(value)/total_date
            ratiovar[date][key] = (ratio, float(value)/threshold)
            if ratio > min_ratio:
                val[key]=ratios[key]
                dates.set_value(date, val)
            if details:
                print("%s %s (%s>%s)" %(date, key, incidents[date], threshold))
                

        if len(incidents) and verbose:
            print("Incidents",key)
            print("[%s - %s] threshold=%s" %(mean, mean+3*std, threshold))
            print(incidents)
            
        if show:
            pylab.title(key+" diffs")
            pylab.plot(diff)
            pylab.plot(diff2)
            pylab.show()
    
    for date, value in dates.items(): 
        dates.set_value(date, ["%s:%.2f->%.2f%%+%.2f" %(k, ratios[k]*100, ratiovar[date][k][0]*100, ratiovar[date][k][1]) for k,v in SortHistogram(value, False, True)])

    ratios = SortHistogram(ratios, False, True)
    return dates

def main(argv=None):
    
    if argv:
        import sys
        sys.argv.extend(argv)
                        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=2,  help="security factor (threshold)")
    parser.add_argument("-f", "--fname", default=None, help="filename")
    parser.add_argument("--quantile", default=None, type=float, 
                        help="quantile to use (not std method; ex: .99)")
    parser.add_argument("--min", default=.01, type=int, help="min percentage")
    parser.add_argument("--details", action="store_true", help="show details")
    parser.add_argument("--verbose", action="store_true", help="show verbose")
    parser.add_argument("--show", action="store_true", help="show diff graph")
    parser.add_argument("--index_col", default="sessionDate", help="index column name")
    parser.add_argument("--sep", default="\t", help="column separator")
    
    args = parser.parse_args()
    if args.fname==None:
        args.fname=fname

    series=pd.read_csv(args.fname, sep=args.sep, index_col=args.index_col)
    print(get_incidents(series, factor=args.factor,quantile=args.quantile,
                        min_ratio=args.min, details=args.details,
                        verbose=args.verbose, show=args.show))
    
if __name__  == "__main__":
    main()
    
    
