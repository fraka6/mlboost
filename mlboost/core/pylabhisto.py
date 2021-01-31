
# Copyright (C) 2006-2009 Francis Pieraut

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


# simple fct to display an histogram sorted 
from pylab import *
from .pphisto import SortHistogram
from numpy import arange,zeros,array,where
from .ppcorr import probs2corr

# format float
def floatformat(x):
    return '%(x).2f' %  {"x": x}

# return x,y,xlabels from hm
def getxyxlabels(hm, n=None):
    # set n
    if n==None:
        n=len(hm)
    else:
	n=min(n,len(hm))
    
    # Specific case
    if type(hm)==type(()):
	return list(range(len(hm[0]))),hm[1],hm[0]
    # else asume it is a freqdist	
	
    # sort histogram	
    shm=SortHistogram(hm,False,True)
    x=arange(n,dtype=float)
    y=zeros(n,float)
    xlabels=[]
    for i in range(n):
	xlabels.append(shm[i][0])
	y[i]=shm[i][1]
    return x,y,xlabels

def getxyxlabelfromref(hm, refhm, n=None,verif=True):
    x,y,xlabels=getxyxlabels(refhm,n)
    if type(hm)==type(()):
	return getxyxlabels(hm,n)
 
    # recompute y for bad and reasign it
    newy=zeros(len(y))
    for i in range(len(xlabels)):
        el=xlabels[i]
	if (el in hm):
	    if verif:
	    	if hm[el]>y[i]:
		    print("problem (",el,")",hm[el],">",y[i])
	    newy[i]=hm[el]
    return x,newy,xlabels
	    
def getxlabelpos(x):
    xlabelpos=array(x)
    xlabelpos+=0.5
    return xlabelpos

# hm is an hashmap or a dictionary
def plothisto(hm, n=10, doshow=True, dolog=False,color='blue'):
    x,y,xlabels=getxyxlabels(hm,n)
    xticks(getxlabelpos(x),xlabels,rotation='vertical')
    bar(x,y,color=color,log=dolog)
    ylabel('counts')
    if doshow:
    	show()
    
def overlaphisto(allhm, badhm, n=10, doshow=True, dolog=False,plotref=True):
    if plotref:
    	plothisto(allhm,n,False)
    x,y,xlabels=getxyxlabelfromref(badhm,allhm,n)
    xticks(getxlabelpos(x),xlabels,rotation='vertical')
    bar(x,y,color='red',log=dolog)
    if doshow:
    	show()


def ShowBucketBinCorr(dists,n=4,corr=None,mincorr=.4):
    # cleanup list from mincorr constraint
    if corr!=None:
	mincorrdist={}
	#scorr=SortHistogram(corr,False,True)
	for el in corr:
	    if corr[el]>=mincorr:
		val=dists.pop(el)	
	    	mincorrdist[el]=val
    	dists=mincorrdist

    print("%s distributions to show" % len(dists))
    # create subplot of n lines 
    while len(dists)>0:
	selecteddists={}
	for i in range(min(n,len(dists))):
	    key,val=dists.popitem()	
	    selecteddists[key]=val
	ShowBinCorr(selecteddists)
	
# dists is an {} of feature containing dist:all,good,bad
# use Dataset.GetBinCorrelation(...)
def ShowBinCorr(dists,nitem=10):
    ncols=6
    doshow=False
    nfeatures=len(dists)
    features=list(dists.keys())
    # subplot idx
    subi=0
    for i in range(nfeatures):
	feature=features[i]
	dist=dists[feature]
	subi+=1
	subplot(nfeatures,ncols,subi)
	plothisto(dist['all'],nitem,doshow=doshow)
	title(feature+" FreqCounts")
	subi+=1
	subplot(nfeatures,ncols,subi)
	overlaphisto(dist['all'],dist['bad'],nitem,doshow=doshow)
	title(feature+" #Bad")
	subi+=1
	subplot(nfeatures,ncols,subi)
	overlaphisto(dist['all'],dist['bad'],nitem,doshow=doshow,plotref=False)
	title(feature+" #Bad (same order)")
	subi+=1
	subplot(nfeatures,ncols,subi)
	plothisto(dist['bad'],nitem,doshow=doshow,color='red')
	title(feature+" #Bad")
	subi+=1
	subplot(nfeatures,ncols,subi)
	sumcorr=overlaphistoprob(dist['all'],dist['bad'],nitem,doshow=doshow)
	title(feature+" error probability")
	subi+=1
	# add correlation info
	x=1-(1.0/(ncols-1))
	y=1.0-(i+1.0)/(nfeatures+1)#y=1.0-2*(i+1.0)/(2*nfeatures+1)
	figtext(x,y,feature+"\nCorrelation="+floatformat(sumcorr*100)+"%")
	
    subplots_adjust(wspace=1,hspace=1)
    show()

def overlaphistoprob(allhm, badhm, n=10, doshow=True, dolog=False):
    x,ay,xlabels=getxyxlabels(allhm,n)
    x,by,xlabels=getxyxlabelfromref(badhm,allhm,n)
    probs=by/ay
    bar(x,probs*100,color='red',log=dolog)
    xticks(getxlabelpos(x),xlabels,rotation='vertical')
    threshold=zeros(len(by))
    threshold[where (probs>=.5)]=50
    bar(x,threshold,color='gray',log=dolog)
    # force ymax to 100%
    axis(ymax=100)
    if doshow:
    	show()
    # recompute probs to consider all buckets YES non nbucket constraint
    x,ay,xlabels=getxyxlabels(allhm)
    x,by,xlabels=getxyxlabelfromref(badhm,allhm)
    probs=by/ay
    probs2corr(probs,by)
    return probs2corr(probs,by)    
	
if __name__ == "__main__":
    hm={'selection': 2, 'locat': 1, 'nois': 1, 'violation': 2, 'text': 1, 'controls': 2, 'should': 37, 'terminat': 1, 'unreliable': 1, 'squeeze': 1, 'fact': 5}
    plothisto(hm)
