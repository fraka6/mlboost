#!/usr/bin/env python
''' provide utils fonction about season, ex date to season '''
from datetime import datetime

def get_season(datein, verbose=False):
    """
    convert date to month and day as integer (md), e.g. 4/21 = 421, 11/17 = 1117, etc.
    """
    if verbose:
        print("datein #1:", datein, type(datein))
    def int2season(s):
        if s==0:
            return "spring"
        elif s==1:
            return "summer"
        elif s==2:
            return "fall"
        elif s==3:
            return "winter"
    
    if isinstance(datein, datetime):
        date = datein.date()
    elif isinstance(datein, str):
        date = datetime.strptime(datein,'%d/%m/%Y').date()
    else:
        raise ValueError('ERROR wrong datein type'+str(type(datein)))
    if verbose:
        print("datein #2:", datein, date)
    
    m = date.month * 100
    d = date.day
    md = m + d

    if (( 80<= md ) and (md <= 171)):
        s = 0 #"spring"
    elif ((172 <= md) and (md <= 265)):
        s = 1 #"summer"
    elif ((266 <= md) and (md <= 354)):
        s = 2 #"fall"
    else:
        s=3 #winter
        
    season = int2season(s)
    if verbose:
        print("get_season is working")

    return season

# test get_season, will generate error if not wroking so don't remove those lines OK 
new_date=datetime(2012,9,16)
get_season(new_date)
get_season("01/02/2019")