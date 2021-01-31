#!/usr/local/bin/python

# ppdist is part of ppboost
# ppboost: PreProcessing boost library 
# ppboost should help you boost your (Machine Learning) Preprocessing steps  
# Copyright (C) 2006-2009  Francis Pieraut

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from .ppcorr import *
from math import *
from collections import defaultdict

# class probability distribution 
def entropy(probs):
    # check that sum=1
    if array(probs).sum()<0.99:
        sys.stderr.writelines("distribution doesn't sum to 1")
    e=0
    for p in probs: 
        if p>0.01: # else=0
            e+=p*log(p,2)+p*log(1-p,2)
    return -e

# entropy of probability p
# WARNING: return -p*log(p) as normaly used for entropy computation
def plog2p(p):
    if p>0.01:
        return -p*log(p,2)
    return 0

# compute the overlap of 2 distributions
def Overlap(dist1,dist2):
        results={}
        for field in list(dist1.keys()):
            if field in dist2:    
                df1=dist1[field]
                df2=dist2[field]
                
		    # continuous overlap
            if ('mean' in df1) & ('var' in df1):
                results[field]=ContinuousOverlap(df1['mean'],df1['var'],df2['mean'],df2['var'],field)
            # categorical overlap
            else:
                results[field]=CategoricalOverlap(df1,df2)
 
        return results

# get difference of 2 categorical distribution
# if sum=True, return the sum of differences over all categories
# otherwise, the hasmap of differences
def CategoricalDiff(df1,df2,sum=True):
    if (isinstance(df1,type({}))==False) & (isinstance(df2,type({}))==False):
        sys.stderr.writelines("wrong parameters in "+sys._getframe().f_code.co_name)
        raise('problem is categoricaldiff arguments')
       
    # sum and map of differences
    sdiff=0.0
    diff={}
    allvals=list(df1.keys())
    allvals.extend(list(df2.keys()))
    vals=uniq(allvals)
    
    for val in vals:
        p1=0.0
        p2=0.0
        if val in df1:
            p1=df1[val]
        if val in df2:
	        p2=df2[val]
        d=abs(p1-p2)
        diff[val]=d
        sdiff+=d

    if sum:
        return sdiff
    else:
        return diff


# overlap of those 2 distributions: mean,stddev (1,2)
# assomption: normal distributed
def ContinuousOverlap(m1,v1,m2,v2,field="?",nsplit=25):
    # get distribution limits points
    pts=[m1-3*v1,m1+3*v1,m2-3*v2,m2+3*v2]
    pts.sort()
    # get junction point
    start=pts[1]
    stop=pts[2]
    step=(stop-start)/nsplit
    x=start-step
    stop=False
    while (stop==False) & (x<=stop):
        x=x+step
        p=[]
        p.append(dist.normcdf(x,m1,v1))
        p.append(dist.normcdf(x,m2,v2))
        p.sort()
        if (p[1]-p[0]<=step):
            stop=True

    if stop==False:
         print("problem !!!!!,set default value: field",field,m1,v1,m2,v2)
         x=(pts[2]-pts[1])/2
         # compute p(X<x)
         return 2*dist.normcdf(x,0,1)
    
    # compute overlap
    x1=x-pts[1]
    x2=pts[2]-x

    if m1<x:
        p=dist.normcdf(x1,0,v1)
        p+=dist.normcdf(x2,0,v2)
    else:
        p=dist.normcdf(x1,0,v2)
        p+=dist.normcdf(x2,0,v1)
	
    return p

# get overlap of those 2 distribution
def CategoricalOverlap(df1,df2):
    overlap=0.0
    for val in list(df1.keys()):
        if val in df2:
            p1=df1[val]
            p2=df2[val]
            overlap+=min(p1,p2)
    return overlap

# input is a list of frequency distributions
def GetDistMergeFromFreqDistList(freqdists, norm=False, uniform=False, perc=False):
    ''' if uniform -> consider uniform weight for each distribution 
        if perc -> generate a percentage '''
    mergedist=defaultdict(lambda:0.0)
    total=0
    for freqdist in freqdists:
        if uniform:
            total+=1
            sumval = sum(freqdist.values())
            for key,val in list(freqdist.items()):
                mergedist[key]+=float(val)/sumval
        else:
            for key,val in list(freqdist.items()):
                total+=val
                mergedist[key]+=val
    # normalize
    if norm or perc:
        total = (1.0*total)/100 if perc else total
        for key in mergedist:
            mergedist[key]/=total
    
    return mergedist    
    
def GetFreqDistListStability(freqdists):
    
    stability=1
    # generate Distribution list
    dists=[]
    for fdist in freqdists:
        #print fdist
        dists.append(GetDistFromFreqDist(fdist))
    
    for i in range(len(dists)-1):
        stability*=CategoricalOverlap(dists[i],dists[i+1])
    return stability
    
def GetDistFromFreqDist(freqdist):
    dist=dict(freqdist)
    sum=0
    for f in freqdist:
        sum+=freqdist[f]
    
    for f in dist:
        dist[f]/=sum
    
    return dist
        
def PError(dist1,dist2):
    o=Overlap(dist1,dist2)
    for el in o:
	    o[el]=100-o[el]
    return o
 
def PercentileFiltering(self,field,pct=.95):
    df=self.data[field]
    pd=PercentileDictionary(df)
    pc=zeros(self.nrec)
    for i in range(self.nrec):
    	pc[i]=pd[df[i]]
    idx=(where(pc<=pct)[0]);
    print(idx)
    print(df[int(idx)])
    print("PercentileFiltering:",len(idx),"/",self.nrec)
	 
def ChunckOverlap(data,n,fname=None):

    nrec=len(data)
    step=int(float(nrec)/n)
    # previous and next distribution
    previous=dataset.data(list(range(0,step)))

    data=zeros((n-1,len(list(previous.keys()))),Float0)
    file=None
    if fname!=None:
        file=open(fname,'w')
        file.writelines(StringifyValues(list(previous.keys())))
                
    for i in range(1,n):
        start=i*step
        stop=start+step
        next=dataset.stats(list(range(start,stop)))
        #print "[",start-n,start,stop,"]"
        overlaps=Overlap(previous,next)
        for j in range(len(overlaps)):
            data[i-1,j]=overlaps[j]
        if file!=None:
            file.writelines(StringifyValues(overlaps))
        previous=next

    fields=list(previous.keys())
    for i in range(len(fields)):
        d=data[:,i]
        m=average(d)
        dif=(d-m)
        stddev=sqrt(sum(dif*dif)/size(d))
        print(fields[i],m,stddev)
        i+=1
            
def SplitDist(self):
    dist1={}
    dist2={}
    mid=int(self.nrec/2)
    for k in self.data:
        dist1[k]=Histogram(self.data[k][0:mid],1)
        dist2[k]=Histogram(self.data[k][mid:int(self.nrec)-1],1)
    return (dist1,dist2)

def Correlation(self,dataset,sorted=True):
    self.ComputeDistribution()
    dataset.ComputeDistribution()
    pe=PError(self.dist,dataset.dist)
    spe=SortHistogram(pe,False,True)
    [dist1,dist2]=self.SplitDist()
    [dist3,dist4]=dataset.SplitDist()
    o1=Overlap(dist1,dist2)
    o2=Overlap(dist3,dist4)
    print("x,Correlation,Confidence,n")
    corrstats={}
    for el in spe:
        field=el[0]
	# default values
    n=0
    o1i=0
    o2i=0
    if field in self.dist:
        n=len(self.dist[field])
    if field in o1:
        o1i=o1[field]
    if field in o2:
        o2i=o2[field]
    print(el,o1i,o2i,n)
    corrstats[field]=(el[1],o1i,o2i,n)
    if sorted:
        return spe
    return corrstats

def RuleSelection(correlation,overlapBefore,overlapDuring,nBefore,threshold=10):
    w=threshold*((100-overlapBefore)+(100-overlapDuring))
    if (correlation>1) & (correlation>w) & (nBefore>1):
        return True
    return False
    
# TODO: generalize for 2 field
def Dependancy(dataset1,dataset2):
    od=Overlap(dataset1,dataset2)
    # create data (merge of 2 dataset)
    d=dict(self.data)

    for f in list(d.keys()):
        d[f].extend(dataset.data[f])
        
    for i in range(len(od)-1):
        # fields names 
        f1=od[i][0]
        f2=od[i+1][0]
        # data
        d1=d[f1]
        d2=d[f2]
        # get distributions good bad
        dg1=self.dist[f1]
        db2=dataset.dist[f2]
        # count overlap
        n=0.0
        # el of d1 that are in d2
        k=0.0
        for j in range(len(d1)):
            if ( (d1[j] in dg1) & (d2[j] in db2) ):
                if dg1[d1[j]]>db2[d2[j]]:
                    n+=1
                    k+=1
                elif d1[j] in dg1:
                    n+=1
        print("dependancy",f1,f2,"=",n/(self.nrec+k)*100,"%")