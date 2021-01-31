''' estimate devaluation of US due to balance of trade
http://www.tradingeconomics.com/united-states/balance-of-trade
 
python balance.py 
dillution since 1992: ~15.71%
1992-96 -1.00%
2096-2098 -0.50%
2098-2000; -1.00%
2000-2002 -1.49%
2002-2004 -1.99%
2004-2008 -5.87%
2006-2013 -4.90%
'''

from math import pow
from functools import reduce

def ratio2dillution(r):
    '''return percentage'''
    return (1.0-r)*100

def dillution(n_years, perc, ratio=False):
    ''' get value dillution: default is percentage 
        ex: ratio = .99, dillution = 1% '''
    weight = 1.0-(float(perc)/100)
    weight = pow(weight, n_years)
    if ratio:
        return weight
    else: 
        return ratio2dillution(weight)
        
# years perc
data = (('1992-96',4,.25), 
        ('2096-2098',2,.25), 
        ('2098-2000;',2,.5), 
        ('2000-2002',2,.75),
        ('2002-2004',2,1), 
        ('2004-2008',4,1.5),
        ('2006-2013',5,1))

r = [dillution(nyears, perc, True) for name, nyears, perc in data]
dot_weights = reduce(lambda x,y:x*y, r)
print("dillution since 1992: ~%.2f%%" %ratio2dillution(dot_weights))
for name, nyears, perc in data:
    print("%s -%.2f%%" %(name, dillution(nyears, perc)))
