#!/usr/bin/env python
""" generate id stats & retention view 
usage example:

cat file.tsv | idstats.py --id appid >file.idstats.tsv
python idstats.py -i file.tsv -o file.idstats.tsv

cool features:
--ndays_stop -> allow you to compute initial stats of all users
--filter     -> allow to filter id base on a window (like a std timeout window)
--ddd        -> add 'day distribution distribution' + distances

"""
from datetime import datetime, timedelta
import csv, sys
from collections import defaultdict
import json   
import logging
import pickle
from numpy import zeros
import logging
from mlboost.core.rtstats import rtstats
from . import geo

# Else large field value throws an exception
csv.field_size_limit(sys.maxsize)

from .tsvreader import NCSTSVReader
from .tsvdictreader import TSVDictReader

def norm(dist):
    new=dist.copy()
    total=sum([float(el) for el in list(dist.values())])
    for key in dist:
        new[key]=float(dist[key])/total*100
    return new

def getSeries(tsvreader, constraint=None):
    '''Returns a generator enumerating all the series found by iterating over 
       tsvreader. contraint = (field,value=None)
       if value != none -> condition is row[id_field]==value 
                  otherwise its a change in id_fields '''
    if constraint:
        field, value = constraint
    previous_value=None
    serie = []

    for row in tsvreader:
        # jump header
        if row[field]==field:
            continue
        # check condition
        if constraint and ((row[field]==value if value else row[field]!=previous_value) 
            and len(serie)>0):
            yield serie
            # create new serie
            serie=[row]
        else:
            serie.append(row)
        previous_value=row[field]
    if len(serie) > 0:
        yield serie

class Global:
    ''' keep track of start and end timestamp & retention mechanism
        - useful to filter out id at the begining and end of the period '''
    start = None
    last = None
    # dayusage[id-str(day)]> n usage that day
    dayusage={}
    
    def __init__(self, idstats, groups_constraint_fname=None):
        self.idstats=idstats 
        self.groups_constraint={}
        self.groups_delta=defaultdict(lambda:0)
        if groups_constraint_fname:
            self.load_group_constraint(groups_constraint_fname)

    def save(self, fname):
        self.save_dayusage(fname)
        fname = "%s.idstats.p" %fname
        f = open(fname, 'wb')
        pickle.dump(len(self.idstats), f)
        for stat in list(self.idstats.values()):
            stat.save(f)
        f.close()
        
    def load(self, fname):
        self.load_dayusage(fname)
        fname = "%s.idstats.p" %fname
        f = open(fname, 'rb')
        n = pickle.load(f)
        for i in range(n):
            idstat = IDStats.load(f)
            self.idstats[idstat.id]=idstat

    def load_group_constraint(self, fname):
        ''' Util fct to load group constraint (useful for retention group analysis)
            format=group_name:function:delta
            ex: trial:idstat.period<=2:0 
            ex: heavy:idstat.period>14 and idstat.used_ratio>.5:14
        '''
        self.groups_constraint={}
        f = open(fname, 'r')
        for line in f.readlines():
            line=line.strip()
            name,fct,delta=line.split(':')
            self.groups_constraint[name]=fct
            self.groups_delta[name]=float(delta)
        f.close()

    @property
    def end(self):
        ''' end is last datetime '''
        return self.last

    def save_dayusage(self, fname):
        fname = "%s.dayusage.p" %fname
        f = open(fname, 'wb')
        pickle.dump(self.dayusage, f)
        pickle.dump(list(self.idstats.keys()), f)
        f.close()
    
    @classmethod
    def load_dayusage(cls, fname):
        fname = "%s.dayusage.p" %fname
        print("loading %s" %fname)
        f = open(fname, 'rb')
        cls.dayusage = pickle.load(f)
        cls.ids = pickle.load(f)
        f.close()

    def used_between(self, idstat, start, end):
        id_start=idstat.start.date()
        id_end=idstat.end.date()
        # basic test about user bounderies
        if (id_start >= start) and (id_start <= end):
            return True
        elif (id_end >= start) and (id_end <= end):
            return True
        # if idstart < start etc.
        for day in (start+timedelta(days=i) for i in range((end-start).days)):
            usageid="%s:%s" %(id,day)
            if usageid in self.dayusage:
                return True
        return False

    def get_period_ids(self, ref_start, ref_end, group=None):
        ''' get ids present in a given period '''
        refids=dict()
        group_constraint=self.groups_constraint[group] if group else ''
        for idstat in list(self.idstats.values()):
            if self.used_between(idstat, ref_start, ref_end):
                if not group or (group and eval(group_constraint)):
                    refids[idstat.id]=idstat
        return refids

    def gen_retention(self, fname, period=1, group=None):
        ''' period=block #of days mode '''
        start=self.start.date()
        # if group, apply the delta is required
        if group:
            delta=self.groups_delta[group]
            start+=timedelta(days=delta)
            if start>=self.end.date():
                logging.warning("can't compute retention;delta too big")

        nblocks = int((self.end.date()-start).days/period)
        
        size_info=[len(self.idstats)]
        retention=zeros((nblocks, nblocks), dtype=float)
        for refi in range(nblocks):
            ref_start=start+timedelta(days=period*refi)
            ref_end=ref_start+timedelta(days=period)
            refids=self.get_period_ids(ref_start, ref_end, group)     
            nref=len(refids)
            size_info.append(nref)
            # don't need to do last row
            if refi==(nblocks-1):
                retention[refi][refi]=100
                continue
            
            # don't need to continue if nref==0
            if nref==0:
                continue
        
            for j in range(refi, nblocks):
                n=0.0
                j_start=start+timedelta(days=period*j)
                j_end=j_start+timedelta(days=period)
                for idstat in list(refids.values()):
                    if self.used_between(idstat, j_start, j_end):
                        n+=1
                retention[refi][j]=100*(n/nref)

        from os import mkdir
        from os.path import isdir
        # save retention matrix
        if not isdir('retention'):
            mkdir('retention')
        fname="retention/%s.retention.%idays.p" %(fname, period)
        if group:
            fname = fname.replace('.p','.%s.p' %group)
        f=open(fname, 'wb')
        pickle.dump(size_info, f)
        pickle.dump(retention, f)
        f.close()

    @classmethod
    def show_retention(cls, fname):
        f = open(fname, 'rb')
        size_info = pickle.load(f)
        retention = pickle.load(f)
        n,n=retention.shape
        print("n id:",size_info[0])
        print("bucket size:",size_info[1:])
        print("bucket perc",["%2.2f%%" %(100*float(val)/size_info[0]) for val in size_info[1:]])
        print("retention:",fname)
        for i in range(n):
            line=["%3.2f" %el for el in retention[i]]
            for i in range(n):
                line[i]=' '*(6-len(line[i]))+line[i]+"%"
            print(line)
        
    def update(self, start, end):
        if self.start==None:
            self.start = start
            self.end = start
        elif end and (end > self.end):
            self.end = end

    def is_valid(self, idstat, min_days):
        ''' return valid(bool) and category (FULL, INCOMPLETE, INCOMPLETE_NEW, 
            INCOMPLETE_NEW_MIN, INCOMPLETE_PRIOR, INCOMPLETE_PRIOR_MIN)'''
        after_start = (idstat.start.date() > self.start.date())
        end_before_end = (idstat.last.date() < self.end.date())
        
        is_min_days = idstat.period >=min_days 
        # automatically validated
        if (after_start and end_before_end):
            return True,"FULL"
        elif (not after_start and not end_before_end):
            return True, "INCOMPLETE"
        elif (after_start and not end_before_end):
            if idstat.period>min_days:
                return True, "INCOMPLETE_NEW_MIN"
            else:
                return False, "INCOMPLETE_NEW"
        elif (not after_start and end_before_end):
            if idstat.period>min_days:
                return True, "INCOMPLETE_PRIOR_MIN"
            else:
                return False, "INCOMPLETE_PRIOR"
        else:
            raise("untreated case")
        

STATS = ['count', 'sum', 'mean', 'stddev']

class AltInfo:
    ''' alternate info class '''
    count=0.0
    previous=None
    def toPerc(self,total):
        return self.count/total

class ConstraintRatio:
    ''' Constraint ratio info class '''
    def __init__(self, num_value, num_key, denom_value, denom_key):
        self.num_value = num_value
        self.num_key = num_key
        self.denom_value = denom_value
        self.denom_key = denom_key
        self.num = defaultdict(lambda:0.0)
        self.denom = defaultdict(lambda:0.0)

    def update(self, id, row):
        if row[self.denom_key]==self.denom_value:
            self.denom[id]+=1
            if row[self.num_key]==self.num_value:
                self.num[id]+=1

    def ratio(self, id):
        num, denom = self.num[id],self.denom[id]
        if denom!=0:
            return str((num,denom,num/denom))
        else:
            return None

    def name(self):
        return "%s_%s_ratio" %(self.num_value,self.denom_value)

class IDStats:
    ''' keep all id stats and feature component 
        WARNING: need to call idStats.config(fields) first !!!
    '''
    stats = STATS
    fields = None
    time_field = None
    dist_fields = None
    # date time start and end
    start = None
    end = None
    
    def __init__(self):
        self.id=None
        self.field_stats = {}
        # create fields stats
        for field in self.fields:
            self.field_stats[field]=rtstats()
        self.start = None
        self.last = None
        self.dist = {}
        for field in self.dist_fields:
            self.dist[field]=defaultdict(lambda:0.0)
        self.ndays = 0
        self.ndays_not_used = 0
        self.hits = 0
        self.speeds = []
        self.no_speed = 0.0
        self.add_location = None
        self.geo = '?'
        self.alt_fields_info=dict((field, AltInfo()) for field in self.alt_fields)
        self.evoln={}
        for field in self.dist_fields_evol:
            self.evoln[field]=[defaultdict(lambda:0.0)]
        self.subset=defaultdict(lambda:0.0)

            
    def extend_evol(self):
        ''' add a bucket to dist_fields_evol dictionary '''
        for field in self.dist_fields_evol:    
            self.evoln[field].append(defaultdict(lambda:0.0))

    def save(self, writer):
        pickle.dump(self.fields, writer)
        pickle.dump(self.dist_fields, writer)
        pickle.dump((self.id, self.start, self.last, self.ndays, 
                     self.ndays_not_used,
                     self.hits, self.speeds, self.geo), writer)
        # field_stats
        for field in self.fields:
            pickle.dump(self.field_stats[field].get_params(), writer)
        # dist 
        for field in self.dist_fields:
            pickle.dump(list(self.dist[field].items()), writer)
    
    @classmethod
    def load(cls, reader):
        new = IDStats()
        new.fields = pickle.load(reader)
        new.dist_fields = pickle.load(reader)
        (new.id, new.start, new.last, new.ndays, 
         new.ndays_not_used, 
         new.hits, new.speeds, new.geo) = pickle.load(reader)
        # field_stats
        for field in new.fields:
            new.field_stats[field] = rtstats.create(*pickle.load(reader))
        # dist 
        for field in new.dist_fields:
            new.dist[field].update(pickle.load(reader))
        return new
                                          
    @classmethod
    def config(cls, fields, id_field, time_field=None, 
               dist_fields=[], alt_fields=[], convert_time_method='tstime', 
               stats=STATS, ndays_stop=None, n_stop=None, ddd=False, 
               position_field=False, active_geo=False,
               dist_step=None, dist_unit='day', dist_fields_evol=[],
               skip=None, subset=None, ratio=None):
        ''' global idStats configuration 
            n_days_stop = usefull option to compare start ndays of 
            different group of users
            ddd = day distribution distance '''
        cls.fields = fields
        cls.id_field = id_field
        cls.time_field = time_field
        cls.dist_fields = dist_fields
        cls.alt_fields = alt_fields
        cls.stats = stats
        cls.convert_time_method = convert_time_method
        cls.ndays_stop = ndays_stop
        cls.n_stop = n_stop
        cls.ddd = ddd
        cls.position_field = position_field
        cls.active_geo = active_geo
        if active_geo:
            cls.location = geo.Location()
        cls.dist_step = dist_step
        cls.dist_fields_evol = dist_fields_evol
        cls.dist_unit = dist_unit
        cls.skip = skip
        cls.subset_params = subset
        cls.ratio = ratio
        cls.did_skipped = False
    
    @classmethod
    def convert_time(cls, string):
        if cls.convert_time_method == 'none':
            return lambda string:string
        # ex: 20140103020708930000
        elif cls.convert_time_method =='tstime':    
            timestr=string[:14]
            if not timestr.isdigit() or " " in timestr:
                logging.warning("<%s> not in appropriate tstime format" %s)
                return
            else:
                try:
                    return datetime.strptime(timestr,"%Y%m%d%H%M%S")
                except Exception as ex:
                    logging.warning(ex)
                    return 

    def to_day_vec(self, norm=False):
        ''' get day vector representation '''
        vec_size = self.period
        vec = zeros(vec_size)
        for i in range(vec_size):
            dayi = self.start+timedelta(days=i)
            usageid="%s:%s" %(self.id,dayi.date())
            vec[i] = Global.dayusage.get(usageid, 0)
        if norm:
            vec/=vec.sum()
        return vec

    def get_day_dist_qty(self, add_vec=False):
        ''' get day distribution quality (uniform = best)
            calculate Kullback-Leibler divergence and Hellinger_distance 
            between normalize day distribution and uniform distribution
           * return the distance vector
        http://en.wikipedia.org/wiki/Kullback-Leibler_divergence
        http://en.wikipedia.org/wiki/Hellinger_distance

        Kullback divergence between 2 vectors (v1 & v2)
        from scipy.stats import entropy
        KL_div = entropy(v1, v2)

        Hellinger Distance between 2 vectors (v1 & v2)
        HD_dist = np.sqrt(0.5 * ((np.sqrt(v1) - np.sqrt(v2))**2).sum())

        '''
        vec = self.to_day_vec(False)
        from scipy.stats import entropy
        import numpy as np
        from numpy import array
        p = vec.sum()/self.period
        uniform = array([p]*self.period)
        info=entropy(vec/vec.sum(), uniform)
        sim = np.sqrt(0.5 * ((np.sqrt(vec) - np.sqrt(uniform))**2).sum())
        if add_vec:
            return ['"%s"' %str(vec).replace('\n',''), info, sim]
        else:
            return [info, sim]

    @property
    def end(self):
        ''' end is last datetime '''
        return self.last

    def update(self, serie):
        # don't update if ndays_stop is activated
        if self.ndays_stop and self.ndays>=self.ndays_stop:
            return True
        if self.n_stop and self.hits>=self.n_stop:
            return True
        
        row = serie[0] 
                
        start = self.convert_time(row[self.time_field])
        last = self.convert_time(serie[-1][self.time_field])
        
        # set id
        if self.id==None:
            self.id=row[self.id_field]
        
        # don't update if in skip list
        if self.skip:
            logging.info("skipping %s" %self.skip)
            for key,val in list(self.skip.items()):
                if row.get(key,None)==val:
                    self.did_skipped = True
                    return True

        # update complex ratios
        if self.ratio:
            for r in self.ratio:
                r.update(self.id, row)

        # update dayusage table
        usageid="%s:%s" %(self.id,start.date())
        Global.dayusage[usageid]=Global.dayusage.get(usageid, 0)+1
        self.hits+=1

        # set default values
        self.speed = '?'
        # measure speed
        if self.position_field:
            if self.hits==1 or self.geo == 'N/A':
                self.geo = row[self.position_field]
                if self.active_geo and self.geo!='N/A':
                    lon1, lat1 = geo.get_long_lat(row[self.position_field])
                    location = self.location.reverse(lon1, lat1)
                    if location and location.raw and 'address' in location.raw:
                        location = location.raw['address']
                        self.geo = str((location.get('city','?'),
                                        location.get('state','?'),
                                        location.get('country','?')))
            speed = 0
            if row[self.position_field] == 'N/A' or serie[-1][self.position_field]=='N/A':
                #logging.warning("first position isn't available %s")
                self.no_speed+=1
            else:
                speed = 0
                lon1, lat1 = geo.get_long_lat(row[self.position_field])
                if len(serie)>1:
                    lon2, lat2 = geo.get_long_lat(serie[-1][self.position_field])
                    speed = geo.speed(lon1, lat1, lon2, lat2, start, last)
                
                if speed>0:
                    self.speeds.append((self.hits, speed))
                    

        # update alt fields count
        if self.alt_fields:
            for field in self.alt_fields:
                alt = self.alt_fields_info[field]
                if row[field]!=alt.previous:
                    alt.count+=1
                    alt.previous=row[field]

        # subset
        if self.subset_params:
            name, field, constraint, keys = self.subset_params
            if row[constraint] in keys:
                self.subset[row[field]]+=1

        # set date
        if self.start is None:
            self.ndays += 1
            self.start = start
            self.last = start
        elif self.start > start:
            logging.warning('Unordered series. Username: %s Timestamps: first %s then %s.' % (self.id, self.start, start))
      
        if self.last is not None and self.last.date() < last.date():
            self.ndays += 1

        # set last date
        self.last = last
        
        # update field distribution
        for field in self.dist:
            self.dist[field][row[field]]+=1
 
        #update field evol distribution
        if self.dist_step:
            if ( (self.dist_unit=='hit' and (self.hits%self.dist_step)==0) or
                 (self.dist_unit=='day' and (self.ndays%self.dist_step)==0) ):
                self.extend_evol()
            for field in self.dist_fields_evol:
                self.evoln[field][-1][row[field]]+=1

        # update field stats
        for field in self.field_stats:
            self.field_stats[field].append(row[field])
    
    @classmethod
    def get_header(cls, added_fields=[]):
        header = ['id', 'start', 'last', 'period', 'ndays' ,'ndays_not_used',
                  'ratio_used', 'hits']
        for field in cls.fields:
            header.extend(['%s_%s' %(field, stat) for stat in cls.stats])
        header.extend(cls.dist_fields)
        header.extend(added_fields)
        if cls.ddd:
            header.extend(['day_vec', 'Kullback-Leibler','hellinger'])
        if cls.position_field:
            header.extend(['speed','no_speed_ratio','geo'])
        if cls.dist_fields_evol:
            header.extend(['%s_evol_%s' %(field, cls.dist_unit) for field in cls.dist_fields_evol])
        if cls.alt_fields:
            header.extend(['%s_alt_count' %(field) for field in cls.alt_fields])
            header.extend(['%s_alt_perc' %(field) for field in cls.alt_fields])
        if cls.subset_params:
            name = "%s_%s" %tuple(cls.subset_params[:2])
            header.extend([name, "NOT_%s" %name, "%s_n" %name, "NOT_%s_n" %name])
        if cls.ratio:
            header.extend([r.name() for r in cls.ratio])

        return header

    def is_valid(self):
        
        valid = ( (self.last is not None) and (self.start is not None) )
        if not valid:
            if self.did_skipped==False:
                logging.warning("UserName stat (aka idstat) %s is invalid. (start=%s;stop=%s)" %(self.id,self.start,self.last))
        else:
            if self.period<1:
                logging.warning("order issue dude in id={id},start={start} last={last}; we are assuming sorted files. You might be using cat *....".format(start=self.start.date(), last=self.last.date(), id=self.id)) 
                valid = False
        return valid
    
    @property
    def period(self):
        return (self.last.date() - self.start.date()).days + 1

    @property
    def ratio_used(self):
        return float(self.ndays)/self.period

    def torow(self, normalize=False):
        """ Recommendation: call self.is_valid() first."""
        # id + start + end time +...
        period = self.period
        row = [self.id, self.start.date(), self.last.date(), period,
               self.ndays, period-self.ndays, self.ratio_used, self.hits]
        # add field stats
        for fieldstat in list(self.field_stats.values()):
            row.extend([fieldstat.n, fieldstat.sum, fieldstat.mean(), fieldstat.stddev()])
        # add distribution
        for field in self.dist_fields:
            fielddist = self.dist[field]
            if normalize:
                total = sum(fielddist.values())
                for k in fielddist:
                    fielddist[k]/=total
            row.append(json.dumps(norm(self.dist[field])))
        return row

def apply(args, input, output):
    ''' arguments are iostreams '''
    #reader = csv.DictReader(input, delimiter="\t", restval='?', quoting=csv.QUOTE_NONE)
    reader = TSVDictReader(NCSTSVReader(input))
    #writer = csv.DictWriter(output, reader.fieldnames, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)

    add_field = args.filter_n_days>0
    new_fields = ['valid','category'] if add_field else []
    idstats=defaultdict(IDStats)
    glob = Global(idstats, args.groups_constraint_fname)
    if args.load_fname:
        glob.load(args.load_fname)

    id_field = args.id_field
    output.write("\t".join(IDStats.get_header(new_fields)) + "\n")
    for serie in getSeries(reader, args.constraint):
        id = serie[0][id_field]
        idstat = idstats[id]
        idstat.update(serie)
        # remove idstat if it was skipped
        if idstat.did_skipped and (idstat.start==None and idstat.last==None):
            del idstats[id]
        else:
            glob.update(idstat.start, idstat.last)
    
    # retention
    if args.retention:
        fname, periods = args.retention.split(':')
        # generate retention matrix for all groups
        groups=list(glob.groups_constraint.keys())
        groups.append(None)
        for group in groups:
            for period in [float(val) for val in periods.split(',')]: 
                glob.gen_retention(fname, period, group)
        glob.save_dayusage(fname)

    if args.save_fname:
        glob.save(args.save_fname)

    for idstat in list(idstats.values()):
        if add_field:
            valid, category = glob.is_valid(idstat, args.filter_n_days)
        if idstat.is_valid():
            row = [str(el) for el in idstat.torow()]
            if add_field:
                row.extend([str(int(valid)), category])
            if args.ddd:
                # FORCE add vector
                row.extend(idstat.get_day_dist_qty(True))
            if args.position_field:
                row.extend([idstat.speeds, idstat.no_speed/idstat.hits, idstat.geo])
            if args.dist_fields_evol:
                row.extend([json.dumps(idstat.evoln[field]) for field in args.dist_fields_evol])
            if args.alt_fields:
                row.extend([idstat.alt_fields_info[field].count for field in args.alt_fields])
                row.extend([idstat.alt_fields_info[field].toPerc(idstat.hits) for field in args.alt_fields])
            if args.subset:
                # generate inverse
                name=args.subset[1]
                inv=idstat.dist[name].copy()
                for k,v in list(idstat.subset.items()):
                    inv[k]-=v
                n_inv=sum(inv.values())
                n_subset=sum(idstat.subset.values())
                row.extend([json.dumps(norm(idstat.subset)), json.dumps(norm(inv)), n_subset, n_inv])
            if args.ratio:
                row.extend([r.ratio(idstat.id) for r in args.ratio])
            output.write("\t".join(["%s" %el for el in row]) + "\n")
        
def main():

    import argparse
    parser = argparse.ArgumentParser(description=str(__doc__),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  
       
    parser.add_argument("-i", dest="input", default=None, help="input filename")
    parser.add_argument("-o", dest="output", default=None, 
                        help="output filename (default = input+.idstats")
    parser.add_argument("-r", dest="retention", default="", 
                        help="retention parameter; format = name:period (periods split by comma)")
    parser.add_argument("--sr", dest="retention_fname", default=None, 
                        help="load and display retention matrix")
    parser.add_argument("--ld", dest="dayusage_fname", default=None, 
                        help="load dayusage fname")
    parser.add_argument('--gc', dest="groups_constraint_fname", default=None,
                        help="group_constraint_fname -> name:constraint(ex:idstat.period>14):delta")
    parser.add_argument("--id", dest="id_field", default='id_field', 
                        help="default id field (default: %(default)s)")
    parser.add_argument("--stats", dest="stats", default=STATS, 
                        help="default stats computed for each fields (default: %(default)s)")
    parser.add_argument('-f', "--fields", dest="fields", default=None,
                        help="comma separared fields to compute stats on; default=all")
    parser.add_argument('--alt', dest="alt_fields", default=[],
                        help="comma separared fields to compute alternate count")
    parser.add_argument('-t', "--time", dest="time_field", default=None,
                        help="timestamp fieldname")
    parser.add_argument('-c', "--contraint", dest="constraint", default=None,
                        help="serie contraint -> field or field=value; ex -c field,value=None")
    parser.add_argument('-d', "--dist", dest="dist_fields", default=[],
                        help="fields for who you want to compute distribution")
    parser.add_argument('--skip', dest="skip", default=None,
                        help="skip series containing key:value; --skip key1:val1,key2:val2,...")
    parser.add_argument('--subset', dest="subset", default=None,
                        help="generate subset of dist field X when Y in keys; --subset NAME:X:Y:key1,key2,key3,...")
    parser.add_argument('--ratio', dest="ratio", default=None,
                        help="generate  ratio; --ratio num:num_key:denom:denom_key,...")
    parser.add_argument("--dist_evol", dest="dist_fields_evol", default=[],
                        help="fields for who you want to compute distribution n:unit:field1,field2,field3,...(unit = hit or day)")
    parser.add_argument('--convert_time_method', dest="time_method", 
                        default = 'tstime',
                        help="fonction use to convert time")
    parser.add_argument('--filter', dest="filter_n_days", default=0, type=int,
                        help="filter n days before and after")
    parser.add_argument('--ddd', dest="ddd", default=False, action="store_true",
                        help="add 'day distribution distance' metrics")
    parser.add_argument('--ndays_stop', dest="ndays_stop", default=None, type=int,
                        help="stop update after ndays; usefull to compare user groups")
    parser.add_argument('--n_stop', dest="n_stop", default=None, type=int,
                        help="stop update after n interaction; usefull to compare user groups")
    parser.add_argument('--speed', dest="position_field", default=None,
                        help="add dialog speed derive from position_field")
    parser.add_argument('--geo', dest="geo", default=False, action="store_true",
                        help="add geo localisation info; WARNING = quite slow")
    parser.add_argument('--save', dest="save_fname", default=None,
                        help="save fname")
    parser.add_argument('--load', dest="load_fname", default=None,
                        help="load fname") 
    #parser.add_argument('--norm', dest="norm", default=None,
    #                    help="normalize distributions")
    args = parser.parse_args()
    if args.retention_fname:
        Global.show_retention(args.retention_fname)
        sys.exit(0)
    if args.dayusage_fname:
        Global.load_dayusage(args.dayusage_fname)
        sys.exit(0)
    if args.fields:
        args.fields=args.fields.split(',')
    else:
        args.fields=[]#reader.fieldnames
    
    if args.alt_fields:
        args.alt_fields=args.alt_fields.split(',')
    
    if args.constraint and '=' in args.constraint:
        args.constraint=args.constraint.split('=')
    else:
        args.constraint=(args.constraint, None)
    
    if args.skip:
        args.skip = dict([keyval.split(':') for keyval in args.skip.split(',')]) 

    if args.subset:
        args.subset = args.subset.split(':')
        # split constraint keys
        args.subset[-1]=args.subset[-1].split(',')

    if args.dist_fields:
        args.dist_fields=args.dist_fields.split(',')

    if args.ratio:
        args.ratio=[ConstraintRatio(*params.split(':')) for params in args.ratio.split(',')]
            
    dist_step = None
    dist_unit = None
    if args.dist_fields_evol:
        dist_step, dist_unit, args.dist_fields_evol=args.dist_fields_evol.split(':')
        dist_step = int(dist_step)
        args.dist_fields_evol=args.dist_fields_evol.split(',')
        
    IDStats.config(args.fields, args.id_field, args.time_field, 
                   args.dist_fields, args.alt_fields, args.time_method, 
                   args.stats, args.ndays_stop, args.n_stop, args.ddd,
                   args.position_field, args.geo,
                   dist_step, dist_unit, args.dist_fields_evol,
                   args.skip, args.subset, args.ratio)

    finput=sys.stdin
    foutput=sys.stdout
    if args.input:
        finput = open(args.input, 'r')
        if args.output==None:
            args.output = args.input.replace('.tsv', '.idstats.tsv')
    if args.output:
        foutput = open(args.output, 'w')

    args(*finput, **foutput)

if __name__ == "__main__":
    main()
