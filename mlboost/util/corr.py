'''compute  P (Class|X) = Sum(P(X=x|Class)/P(X=x))*p(Class)'''
import sys
from csv import DictReader
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging 

from taskstats.statsgatherer import merge_stats
from taskstats.utils import get_split_by_directory_name
from taskstats.utils import SPLIT_BY_FIELDNAMES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--classe", default='Accepted', help="class column name")
parser.add_argument("--other", default=None, help="default is all expect class and x")
parser.add_argument("-x", default=0, help="x column position")
args = parser.parse_args()
reader = DictReader(sys.stdin, delimiter='\t')

details = False
verbose = True
norm = False
d = []
keys=[]
key_name=reader.fieldnames[args.x]
others = [el for el in reader.fieldnames if el not in (key_name, args.classe)]
for row in reader:
    x = float(row[args.classe])
    y = float(sum(float(row[name]) for name in others))
    d.append([x, y])
    keys.append(row[key_name])

d = np.array(d)
prob_accepted=d[:,0].sum()/d.sum()
if norm:
    d[:,0]/=d[:,0].sum()
    d[:,1]/=d[:,1].sum()

prob_x_accepted=d[:,0]/d[:,0].sum()
prob_x_cancelled=d[:,1]/d[:,1].sum()
prob_x=d.sum(axis=1)/d.sum()
prob_error=d.min(axis=1).sum()/d.sum()
prob_accepted_x=(prob_x_accepted/prob_x)*prob_accepted

if details:
    print(d)
    print("prob_x_accepted:",prob_x_accepted)
    print("prob_x         :",prob_x)
    print("prob_accepted  :",prob_accepted)
    print("p_x_a*p_x      :",prob_x_accepted/prob_x)
    print("p_x_a*p_x*p_a  :",prob_x_accepted/prob_x*prob_accepted)
    print("p_accepted     :",(prob_x_accepted*prob_accepted).sum())
    
if verbose:
    #print "prob_accepted  :%.2f%%" %(prob_accepted*100)
    #print "p(Class=class|x)=%.2f%%" %((1-prob_error)*100)
    print("{0}\t{1}\t{2}".format('Value', 'Probability', 'Proportion'))
    for key,prob,prop in zip(keys, prob_accepted_x, prob_x):
        print("{0:<10}\t{1:.2f}\t{2:.2f}".format(key, prob*100, prop*100))
    print("{0}\t{1:.2f}\t{2:.2f}".format('any', prob_accepted*100, 100))
else:
    print(prob_accepted_x)
