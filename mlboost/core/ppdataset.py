#!/usr/bin/python
# -*- coding: utf-8 -*-

# ppdataset is part of ppboost
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


import os, sys
import re
import time
from .ppcorr import probs2corr
from .pputil import *
from math import *
import csv
from numpy import *
from .pphisto import Histogram, SortHistogram
from .ppdist import *
from .__init__ import extract_features
verbose = False

# IMPORTANT; to avoid matching rounding error in mapping hashing shoud return a float
# because it is stored as a float in the data matrix
old_hash = hash
def hash(value):
    return float(old_hash(value))
    
class Dataset:
    def __init__(self,filename,exception_continuous_fields=[],exception_categorical_fields=[]):
        
        # open file and load header
        self.fname=filename
        file=open(filename,'r')
        self.header=file.readline().strip()
        
        # set fields separator, default =" "
        self.filesep=" "
        if filename.find(".csv")>=0:
            self.filesep=","
        else:
            print("default fields separator=\" \"")

        self.fields=self.header.split(self.filesep)
        self.nfields=len(self.fields)
        # set field type (continuous=1;categorical=0)
        self.AutomaticSetFieldsType(exception_continuous_fields,exception_categorical_fields)
        self.LoadOptimalStructure()
        # number of uniq values/number of examples for a given field 
        self.uniqvalcountratio={}
            
    def PercentileFiltering(self,field,pct=95):
        d=self.GetFieldData(field)
        pd=PercentileDictionary(d)
        pc=zeros(self.nrec)
        # set each value percentile
        for i in range(self.nrec):
            pc[i]=pd[d[i]]
        
        idx=(where(pc<=pct))[0]
       
        if len(idx)>0:
            return take(self.data, idx, axis=0)
        else:
            sys.stderr.writelines(sys._getframe().f_code.co_name+"(): no elements\n")
            raise("no value")
	
    # give a list of constraints(field,value)
    def GetConstrainedData(self,constraints):
        
        if isinstance(constraints,list)==False:
            constraints=[constraints]
        
        # get a copy of data
        data=array(self.data)
        # loop over constraints
        for c in constraints:
        
            # constraint checks
            if len(c)!=2:
                sys.stderr.writelines(sys._getframe().f_code.co_name+" :bad constraint format "+str(c)+"\n")
                continue
        
            # get back field and value constraint
            field,value=c
            self.IsField(field)
            
            if verbose:
                print("data constraint:",c)
            
            # get data and idx
            d=self.GetFieldData(field,data)
            if self.IsCategorical(field):
                value=hash(value)
           
            idx=(where (d==value))[0]
            
            if len(idx)>0:
                data=take(data, idx, axis=0)
            else:
                sys.stderr.writelines(sys._getframe().f_code.co_name+"(): no elements\n")
                raise("empty data")
        
        if verbose:
            print("remaining data:",float(len(data))/len(self.data)) 
        return data
         
    # exception continuous fields  : continuous set to categorical
    # exception categorical fields : categorical set to continuous
    def AutomaticSetFieldsType(self,exception_continuous_fields=[],exception_categorical_fields=[]):
        
        # get first record values
        file=open(self.fname,'r')
        # jump over header
        file.readline()
        values=file.readline().split(self.filesep)
        
        # intialisation of continuous fields names list
        self.continuousfields=[]
        # fieldstype:1=continuous,0=categories
        self.fieldstype=zeros(self.nfields)
        
        for i in range(self.nfields):
            field=self.fields[i]
            val=values[i]
            if field not in exception_continuous_fields:
                try:
                    float(val)
                    self.continuousfields.append(field)
                    self.fieldstype[self.Idx(field)]=1
                except:
                    print(field," is a categorical field")
                    pass
        
         
    # fieldtype: 1 = continuous otherwise categorical 
    # classfield data won't be return if classfieldname!=None 
    def Fields(self):
        return self.fields
    
    def IsField(self,field):
        fcount=self.fields.count(field)
        if fcount==0:
            return False
        elif fcount==1:
            return True
        else:
            sys.stderr("There is "+fcount+" fields")
            return True
        
    def Idx(self,fields):
        idx=[]
        if isinstance(fields,list)==False:
            fields=[fields]
        for field in fields:
            if self.fields.count(field)==0:
                print("*****",self.fields,field)
                sys.stderr.writelines('%s:unknown field:%s\nChoices=%s' %( 
                                      sys._getframe().f_code.co_name,field,self.fields))
            else:
                idx.append(self.fields.index(field))
        
        if len(idx)==0:
            sys.stderr.writelines(sys._getframe().f_code.co_name+" Error:no idx "+str(fields)+"\n")
            sys.exit(1)
        if len(idx)==1:
            return idx[0]
        
        return idx

    def GetFieldData(self,fields,data=None):

        data=self.GetData(data)
            
        if isinstance(fields,list)==True:
            return take(data, self.Idx(fields), axis=1)
        else:
            #print fields,"idx:",self.Idx(fields)
            return data[:,self.Idx(fields)]

    def BucketizeField(self,field,thresholds,data=None):

        if self.IsCategorical(field):
            raise "Warning: "
        data=self.GetData(data)

        # get data
        d=self.GetFieldData(field)
        idx=self.Idx(field)
        nt=len(thresholds)

        # convert data according to thresholds
        for i in range(self.nrec):

            x=d[i]
            cat=0
            for ti in range(nt):
                if x>thresholds[ti]:
                    cat=ti+1
            data[i,idx]=cat
            #print x,cat

        # change mapping
        mapfield={}
        for i in range(nt):
            mapfield[i]="< "+str(thresholds[i])
            #print i,mapfield[i]
        mapfield[nt]="> "+str(thresholds[nt-1])
        #print nt,mapfield[nt]

        
        print("mapfield:",mapfield)
        self.cfhm[field]=mapfield
        self.continuousfields.remove(field)
        self.fieldstype[idx]=0
                    
    # cmptype = -1,0,1, None->lower,equal,higher,not equal
    def GetFieldValueIdx(self, field, value, cmptype=0, data=None):
        data=self.GetData(data)
        # categorical field
        if self.IsCategorical(field):
            if cmptype==0:
                return where (self.GetFieldData(field,data)==hash(value))[0]
            elif cmptype==None:
                return where (self.GetFieldData(field,data)!=hash(value))[0]
            else:
                print("Wrong usage of GetContinuousFielValueIdx for categorical values")
        else:
            # continuous fields    	
            if cmptype==0:
                return where (self.GetFieldData(field,data)==value)[0]
            elif cmptype<0:
                return where (self.GetFieldData(field,data)<value)[0]
            elif cmptype>0:
                return where (self.GetFieldData(field,data)>=value)[0]
        
    # util fct for data assignation
    def GetData(self,data=None):
        if data==None:
            return self.data
        return data
    
    # get a subset of the dataset based on a constraint
    def GetSubset(self, field, value, cmptype=0,data=None):
        data=self.GetData(data)
        cidx=self.GetFieldValueIdx(field, value, cmptype, data)
        return data[cidx,:]
	
    def GetSplitSubsets(self, field, value,reverse=False,data=None):
        data=self.GetData(data)
        if self.IsCategorical(field):
            ds1=self.GetSubset(field,value,0,data)
            ds2=self.GetSubset(field,value,None,data)
        else:
            ds1=self.GetSubset(field,value,1,data)
            ds2=self.GetSubset(field,value,-1,data)
        if reverse:
            return ds2,ds1
            return ds1, ds2     
    
    # warning data colun shouldn't have been alterated
    def GetFreqCounts(self,field,data=None,nbins=100):
        data=self.GetData(data)
        fdata=self.data[:,self.Idx(field)]   
        if self.IsCategorical(field):
            return Histogram(fdata,True)
        else:
            n, bins=histogram(fdata,nbins)
            return n
    
    # warning, use carefully, it is a hack	
    def forcesamesplit(self,d1,d2):
        if (len(d1)==0) | (len(d2)==0):
            return d1,d2
        # force same split 
        fmin=min([min(d1),min(d2)])
        fmax=max([max(d1),max(d2)])
        fminmax=zeros(2)
        fminmax[0]=fmin;fminmax[1]=fmax
        newd1=zeros(d1.size+2)
        newd2=zeros(d2.size+2)
        newd1[0:2]=fminmax;newd2[0:2]=fminmax
        newd1[2:]=d1;newd2[2:]=d2
        return newd1,newd2
    
    # warning, use carefully, it is hack
    def removeminmax(self,ngood,nbad):
	    #print  ngood[0],nbad[0],ngood[-1],nbad[-1]
        ngood[0]-=1;nbad[0]-=1
        ngood[-1]-=1;nbad[-1]-=1
        return ngood,nbad
    
    # useful for categorical histogram
    def RemapHistogram(self,h,field):
        newh={}
        for key in h:
            newh[self.ConvertValue(field,key,"?")]=h[key]
        return newh
        
    def GetXY(self, xfield, yfield, classfield):
        ''' get x and y values for each class; assume classfield is a category '''
        class d:
            pass
        data={}
        # loop over class values
        for  value in self.cfhm[classfield]:
            cvalue = self.ConvertValue(classfield, value)
            data_i=d()
            vdata = self.GetConstrainedData((classfield,cvalue))
            data_i.x = self.GetFieldData(xfield,vdata) 
            data_i.y = self.GetFieldData(yfield,vdata)
            data["%s %s" %(classfield,cvalue)] = data_i
        return data

    # return distributions [all,bin1,bin2]
    def GetBinDists(self,field, value, nbins=25,reverse=True,idx=None,verbose=False, normalize=False):
        if idx==None:
            data=self.data
        else:
            data=self.data[idx,:]
        # old probability
        prob={};
        # FP correlation
        corr={};
        dists={}
        # warning: don't remove field (don't use rmclassfield)
        ds1,ds2=self.GetSplitSubsets(field,value,reverse=True,data=data)
        minel=max(ds2.size,1) # confusion ... max([min([ds1.size,ds2.size]),1])
        if verbose:
            print("good, bad dist:=",float(ds1.size)/data.size*100,"%",float(ds2.size)/data.size*100,"% from",data.size)
        #print "split:",ds1.size(),ds2.size()
        
        # loop over all field except split criterion
        fields=list(self.Fields())
        fields.remove(field)
        for key in fields:
            prob[key]=0
            corr[key]=0
            noverlap=0.0
            dists[key]={}
            fds=self.GetFieldData(key,data)
            fds1=ds1[:,self.Idx(key)]
            fds2=ds2[:,self.Idx(key)]
            n1 = len(fds1)
            n2 = len(fds2)
            normalize_ratio=max(n1,n2)/min(n1,n2)
            if normalize:
                print("normalization ratio %s" % normalize_ratio)
            if round(fds.sum())!=round((fds1.sum()+fds2.sum())):
                print(fds.sum(),'!=',fds1.sum(),"+",fds2.sum(),"but =",(fds1.sum()+fds2.sum()))
                raise("sum doesn't align")
            
            # get freq count dist
            if self.IsCategorical(key):
                all=self.RemapHistogram(Histogram(fds),key)
                probs = zeros(len(all),float)
                nbad = zeros(len(all))
                h1 = Histogram(fds1)
                h2 = Histogram(fds2)
                        
                # scale lower class and update all distribution
                def normalize_histo(histo):
                    for key in histo:
                        all[key]+=hist[key]*(normalize_ratio-1)
                        histo[key]*=normalize_ratio             
                    if normalize:
                        if n1<n2:
                            normalise_histo(h1)
                        else:
                            normalize_histo(h2)

                good=self.RemapHistogram(h1,key)
                bad=self.RemapHistogram(h2,key)
                dists[key]['all']=all
                dists[key]['good']=good
                dists[key]['bad']=bad
            
            # compute probs 
            i=0
            for el in all:
                if el in bad:
                    probs[i]=float(bad[el])/all[el]
                    nbad[i]=bad[el] 
                    i+=1	
            
            # compute overlap
            for el in good:
                if (el in bad):
                    # the overlap is sum of min value (i.e=2*min of divide by the sum of elements)
                    noverlap+=min([good[el],bad[el]])
            
                else: #continuous
                    fds1,fds2=self.forcesamesplit(fds1,fds2)
                    nall, abins=histogram(fds,nbins)
                    ngood=zeros(len(nall));gbins=array(abins)
                    if fds1.size>0:
                        ngood, gbins=histogram(fds1,nbins)
                    nbad=zeros(len(nall));bbins=array(abins)
                    if fds2.size>0:
                        nbad, bbins=histogram(fds2,nbins)    
                
                    def normalize_histo(values):
                        for i in range(len(values)):
                            nall[i]+=values[i]*(normalize_ratio-1)
                            values[i]*=normalize_ratio
                                
                        n1 = len(fds1)
                        n2 = len(fds2)
                        if normalize:
                            if n1<n2:
                                normalize(fds1)
                            else:
                                normalize(fds2)

            # remove min and max added by forcesamesplit
            if (len(fds1)>0) & (len(fds2)>0):
                ngood,nbad=self.removeminmax(ngood,nbad)
            
            # check
            if nall.sum()!=(ngood.sum()+nbad.sum()):
                print(nall.sum(),'!=',ngood.sum(),"+",nbad.sum(),"but =",(ngood.sum()+nbad.sum()))
                print(fds.size(),fds1.size(),fds2.size())
                raise "bins sum doesn't align "
                if ((abins!=gbins).sum()!=0) | ((bbins!=gbins).sum()!=0):
                    raise("force samesplit problem bins")
                # TODO : reactivate !!
                #if ((ngood>nall).sum()!=0) | ((nbad>nall).sum()!=0):
                #    print (ngood>nall).sum(),(nbad>nall).sum()
                #    print (ngood>nall),"\n",ngood,"\n",nall
                #    print (nbad>nall),"\n",nbad,"\n",nall
                #    raise("force samesplit problem counts")
                        
                        # TODO: fix bug + rm hack [:-1]
                dists[key]['all']=(abins[:-1],nall)
                dists[key]['good']=(gbins[:-1],ngood)
                dists[key]['bad']=(bbins[:-1],nbad)
                
                # compute probability
                probs=nbad/nall
                
                # compute overlap 
                for i in range(len(ngood)):
                    noverlap+=min([ngood[i],nbad[i]])
                
                prob[key]=(1-(noverlap/minel))*100
                corr[key]=probs2corr(probs,nbad)
                # missing info
                if (fds1.size==0) | (fds2.size==0):
                    corr[key]=-1
        
        # print sorted corr
        if verbose:
            scorr=SortHistogram(corr,False,True)
            for el in scorr:
                print(el[0],el[1]*100,"% ")
        
        return dists,corr,ds1.size,ds2.size
        
    # TODO: generalize to remove a list of classfieldnames
    def GetDataType(self,fieldtype,rmclassfieldname=None):
        # get field type index        
        idx=(where (self.fieldstype==fieldtype))[0]
        
        # remove classfieldname
        if self.fields.count(rmclassfieldname)>0:
            idx.remove(self.fields.index(rmclassfieldname))
        return take(self.data, idx, axis=1)
    
    def GetCategoricalData(self,rmclassfieldname=None):
        return self.GetDataType(0,rmclassfieldname)

    def GetContinuousData(self,rmclassfieldname=None):
        return self.GetDataType(1,rmclassfieldname)

    # use csv reader lib
    def LoadOptimalStructure(self,sep=","):
        self.n=nlines(self.fname)-1
        starttime=time.clock()
        # open file 
        file=open(self.fname,'r')
        reader = csv.reader(file,delimiter=self.filesep)
        
        # skip header
        file.readline()
        
        # categocial field hashmap : cfhm[field]->hashmap
        # fields hashmap idx : hmidx[i]->hashmap (need for loading optimization)
        self.cfhm={}                
        self.hmidx=[]
        for i in range(self.nfields):
            self.hmidx.append({})
            field=self.fields[i]
            if self.fieldstype[i]==0:
                self.cfhm[field]=self.hmidx[i]
        
        self.nrec=nlines(self.fname)-1
        self.data=zeros((self.nrec,self.nfields),float)
        # convert every lines
        i=0
        for values in reader:
            if (i%1000)==0:
                print("loading %.2s%% %s sec" % (float(i)/self.nrec*100,time.clock()-starttime))
            if len(values)!=self.nfields:
                sys.stderr.writelines(sys._getframe().f_code.co_name+": problem with line "+str(i)+" ("+str(len(values))+"!="+str(self.nfields)+")\n")
                print(values)
                sys.exit(1)
            else:
                isstring=isinstance(values[0],str)
                for j in range(self.nfields):
                    val=values[j]
                    if self.fieldstype[j]==1:
                        try:
                            self.data[i,j]=float(val)
                        except:
                            self.data[i,j]=-1
                    else: # categories
                        # hash value
                        if isstring:
                            val=val.replace(',',' ')
                            val=val.lower()
                        hv=hash(val)
                        if (hv in self.hmidx[j])==False:
                            self.hmidx[j][hv]=val
                        self.data[i,j]=hv
            i+=1
                    
        print("loading time: ",time.clock()-starttime,'sec')
        
    def Print(self):
        for i in range(self.nrec):
            print(self.GetInputLine(i))            

    def ConvertValues(self,field,values):
        cvalues=[]
        for val in values:
            cvalues.append(self.ConvertValue(field,val))
        return cvalues
        
    def ConvertValue(self,field,value,unknown=None):
        
        if self.IsCategorical(field):
            if value in self.cfhm[field]:
                #print value, self.cfhm[field],self.cfhm[field][value]
                return self.cfhm[field][value]
            elif unknown!=None:
                return unknown
            else:
                choices=str(list(self.cfhm[field].keys()))
                raise 'unknown value mapping for '
        else:
            print("Warning: ConvertValue isn't needed")
            return value
        
    def GetInputLine(self,i):
        values=[]
        for j in range(self.nfields):
            if self.fieldstype[j]==1:
                values.append(self.data[i,j])
            else:
                try:
                    values.append(self.hmidx[j][self.data[i,j]])
                except:
                    print("Unknown key :",self.data[i,j]) 
        
        return values
                            
    def Convert(self,sep=","):

        # create output files
        catfname=self.fname+".categories.hash.csv"
        cfname=self.fname+".continuous.csv"
        self.fnames=[catfname,cfname]
        
        self.catf=open(catfname,'w')
        self.cf=open(cfname,'w')
        
        # convert file
        file=open(self.fname,'r')
        file.readline()
        
        # categorical fields hashmap: cffhm[field]->[file,hashmap]
        cffhm={}
        
        # recreate appropriates files header
        for i in range(self.nfields):
            field=self.fields[i]
            
            if self.fieldstype[i]==1:
                self.cf.writelines(field+sep)
            else:# categorical
                self.catf.writelines(field+sep)
                cffhm[field]=(open(self.fname+"."+field+".categories.hashmap.txt",'w'),{})
                # add continuous fields
                self.cf.writelines(field+"-length"+sep)
                
        self.catf.writelines("\n")
        self.cf.writelines("\n")

        # write to files
        for line in file.readlines():
            line=line[0:len(line)-1]
            values=line.split(self.filesep)
            for i in range(self.nfields):
                val=values[i].replace('"',"")
                field=self.fields[i]
                if self.fieldstype[i]==1:
                    # exception value 
                    if val=="-":
                        val="0"
                    self.cf.writelines(val+sep)
                else: # categories
                    # hashmap field
                    hmf=cffhm[field][1]
                    hf=str(hash(val))
                    if (hf in hmf)==False:
                        hmf[hf]=val
                    self.catf.writelines(hf+sep)
                    # add continuous fields
                    self.cf.writelines(str(len(val))+sep)
                    
            self.catf.writelines("\n")
            self.cf.writelines("\n")

        # fill hashmap fields files
        for x in cffhm:
            f,hm=cffhm[x]
            for k in hm:
                f.writelines(str(k)+" "+str(hm[k])+"\n")
        print("files:",str(self.fnames))
    
    def ExtractFeatures(self, fcts, classfield, fname, return_dataset=True, feature_names=None):
        f = open(fname, 'w')
        # get new input size
        header = []
        if not isinstance(fcts, list):
            fcts = [fcts]
        classidx = self.fields.index(classfield)
        header = feature_names or []
        if not feature_names:
            for fct in fcts:
                header.extend(fct(self.data[0]))
                header = list(range(len(header)))
        header.append(classfield)
        
        f.writelines(StringifyValues(header).replace(" ",""))
        for i in range(self.n):
            inputs = extract_features(self.data[i, :-1], fcts)
            # add class
            inputs.append(self.ConvertValue(classfield, self.data[i][classidx]))
            f.writelines(StringifyValues(inputs).replace(" ",""))
        f.close()
        print(fname,"created")
        if return_dataset:
            return Dataset(fname,exception_continuous_fields=[classfield] )

    def GenFlayerFormat(self,classfield, fname=None, convert=True):

        m=re.match('(.*)\.[^.]*',self.fname)
        flayer_fname=(fname or m.group(1))+".data"
        fout=open(flayer_fname,'w')
        # compute number of inputs
        # categories field will be convert into inputs if number of distinct values is <= maxcat
        ninputs=str(self.nfields-1)
        classidx=self.fields.index(classfield)
        idx=list(range(self.nfields))
        idx.remove(classidx)
        nclasses=str(len(self.cfhm[classfield]))
        fout.writelines(str(self.nrec)+" "+ninputs+" "+nclasses+"\n")
        #convert = self.IsCategorical(classfield) and convert
        print("class idx %s" % classidx)
        for i in range(self.nrec):
            features=self.data[i,idx]
            target=self.data[i][classidx]
            fout.writelines(str(features).replace('[','').replace(']',''))
            if convert:
                try:
                    target=self.ConvertValue(classfield, target)
                except:
                    print("problem convertion at line#",i)
            fout.writelines(' '+str(target)+'\n')
        print("%s created" %flayer_fname) 
        fout.close()
                
    def IsContinuous(self,field):
        return (self.fieldstype[self.Idx(field)]==1)

    def IsCategorical(self,field):
        return (self.fieldstype[self.Idx(field)]==0)
    
    def GetContinuousIdx(self):
        return (where (self.fieldstype==1))[0]
        
    def GetCategoricalIdx(self):
        return (where (self.fieldstype!=1))[0]
    
    def GetDist(self,fields=None,idx=None):
        if idx==None:
            idx=list(range(self.nrec))
        stats={}
        
        if fields==None:
            # continuous
            coidx=self.GetContinuousIdx()
            # categories
            caidx=self.GetCategoricalIdx()
        else:
            if isinstance(fields,list)==False:
                fields=[fields]
            coidx=[]
            caidx=[]
            for field in fields:
                fidx=self.Idx(field)
                if self.IsCategorical(field):
                    caidx.append(fidx)
                else:
                    coidx.append(fidx)
                
        # continuous
        for i in coidx:
            fdata=self.data[idx,i]
            field=self.fields[i]
            dist={}
            dist["mean"]=mean(fdata)
            dist["stddev"]=std(fdata)
            dist["median"]=median(fdata)
            stats[field]=dist
        
        # categorical fields
        for i in caidx:
            stats[self.fields[i]]=Histogram(self.data[idx,i],True)
        
        if len(stats)==1:
            return stats[list(stats.keys())[0]]
        return stats
        
    def FreqDist(self,field,minvalues=30,maxvalues=50,data=None,save=False,reset=True):
        
        if self.IsCategorical(field):
            sys.stderr(field+" isn't a continuous values. You can't call:"+sys._getframe().f_code.co_name)
            return self.GetDist(field)
        
        if reset:
            gnuplot.__call__("reset")                     
            
        data=self.GetData(data)
        
        fdata=self.GetFieldData(field,data)
        h=Histogram(fdata,True)
        n=len(h)
        # exception buckets
        k=maxvalues
        if n>k:
            n=k
        # if the number of different values is too small, consider it as a categorical field
        elif n<minvalues:
            sys.stderr.writelines(sys._getframe().f_code.co_name+": continuous "+field+" add only "+str(n)+" values, \n")
            self.PlotHistogram(h,field)
            #ylabel('%')
            return 
       
        # count el by buckets
        sd=array(fdata)
        sd.sort()
        fmin=sd[0]
        fmax=sd[-1]
        stop=fmin
        step=(fmax-fmin)/n
        print(fmin,fmax,step)
        # bucket count
        bc=zeros(n,'Float')
        xlabels=[]
        for i in range(n):
            start=stop
            stop=start+step
            xlabels.append(str(start+.5*step))
            bc[i]=((fdata>start)&(fdata<stop)).sum()
            #print start,stop,bc[i]
        
        bc=bc/bc.sum()*100
        
        x=list(range(n))
        y=bc
        return x,y,xlabels
     
    def SplitDist(self,fields=None):
        mid=int(self.nrec/2)
        return ( self.GetDist(fields,list(range(0,mid))) ,self.GetDist(fields,list(range(mid,int(self.nrec)-1))) )
    
                        
    def PrintDist(self,field):
        d=self.GetDist()
        i=1
        for el in d:
            print(i,el[0],el[1],"%") 
    
    # This function return field entropy and probability of good classification 
    # if we use that feature as the only predictor 
    # binary problem entropy H=-(p*log(p)+(1-p)*log(1-p))
    # see Shannon paper from 1948:
    # http://cm.bell-labs.com/cm/ms/what/shannonday/shannon1948.pdf
    # WARNING: all possible class field values will be considered as different classes    
    def InfoProb(self, field, classfield, computeclassprob=False):
        # get class values
        classvalues=self.GetFieldData(classfield)
        # get list possible class values
        possibleclassvalues=uniq(classvalues)
       
        info=0.0 # information contained by this field related to classfield classification capacity
        classprob=1 # classification accuracy if we split on the given field
        
        # default values for information theoretic and classification probability threashold 
        vi=0
        vc=0
           
        # categorical ?
        if self.IsCategorical(field):
            dist=Histogram(self.GetFieldData(field),True)
            classprob=self.GetClassificationAccuracy(field,classfield)
            self.uniqvalcountratio[field]=float(len(dist))/self.nrec
            if 0:
                print(field,len(dist),self.nrec,self.uniqvalcountratio[field])
                
            for value in list(dist.keys()):
                w=dist[value]/100
                idx=self.GetFieldValueIdx(field,value)
               
                if len(idx)>0:
                    # sum of probability used for verification: sum(x)=1
                    sumprob=0
                    # for all possible class value
                    for classvalue in possibleclassvalues:
                        n=(classvalues[idx]==classvalue).sum()
                        p=float(n)/len(idx)
                        info+=w*plog2p(p)
                        classprob-=w*p
                        sumprob+=p
                       
                    if sumprob <0.99:
                        sys.stderr.writelines("Problem: sum possible class probabilty doesn't sum to 1")
                   
        else: # continuous values
            info=1
            classprob=0
        
            d=self.GetFieldData(field)
            self.uniqvalcountratio[field]=2.0/self.nrec # only 2 possible choces: < or > than the threshold 
            # get sorted possible values
            sv=uniq(d)
            sv.sort()
            # previous value
            pv=sv[0]
            # compute info for each possible values (optimized version)
            infov={}
            # compute classification accuracy for each value, what we are really optimizing
            classprobv={}
            # compute class distribution
            classdist=Histogram(classvalues,True)
            # store cumulative probability and initialize values 
            classcumprob={}
            for classval in possibleclassvalues:
                classcumprob[classval]=0
                
            # compute info for all values (except first one, optimization)
            for v in sv[1:]:
                infov[v]=0.0
                # usefull to verify that entropy is a good cost function...
                if computeclassprob:
                    classprobv[v]=self.GetClassificationAccuracy(field,classfield,v)
                   
                # get new idx 
                idx=(where ( (d>pv) & (d<=v) ))[0]
                
                # for all possible class field values (update version):
                for classval in possibleclassvalues:
                    n=(classvalues[idx]==classval).sum()
                    classcumprob[classval]+=float(n)/self.nrec
                    infov[v]+=plog2p(classcumprob[classval])
                    #infov[v]+=float(classdist[classval])/100*plog2p(classcumprob[classval])
                    #print classvalue,p,plog2p(p),float(classdist[classvalue])/100,infov[v]
                    
                # keep previous value, DON'T REMOVE THAT LINE
                pv=v
            
            #get highest info value (first one)...Yes we are computing 1-entropy
            if len(infov)>=1:
                [vi,info]=SortHistogram(infov,False,False)[0]
              
            if 0:
                print(field,"threshold:",vi,"Classification:",self.GetClassificationAccuracy(field,classfield,vi),"entropy:",info)
                
            if computeclassprob:
                #get max classification value (first one)
                if len(infov)>=1:
                    [vc,classprob]=SortHistogram(classprobv,False,True)[0]
            
        if 0:
            print("field,info,prob")
            print(field,info*100,classprob*100)
            print("value: v",vi,"entropy:",info,"classification:",self.GetClassificationAccuracy(field,classfield,vi))
            if computeclassprob:
                print("best classification:",v,self.GetClassificationAccuracy(field,classfield,vc))
               
        return (1-info),vi,classprob,vc
    
    # classfield need to be binary
    def GetFeatureErrorClassification(self,classfield):
    
        if self.fields.count(classfield)!=1:
            sys.stderr.writelines('unknown class field :'+classfield+'\n')
            print(self.fields)
            return
        
        fields=list(self.fields)    
        fields.remove(classfield)
        # TODO ...finish it please
        #for field in fields:
            # Get distribution classes distribution   
            #get 
        
        
    # for continuous field, threshold need to be set
    def GetClassificationAccuracy(self,field,classfield,threshold=None):
        
        classvalues=self.GetFieldData(classfield)
        # get list possible class values
        possibleclassvalues=uniq(classvalues)
        # get field data  
        d=self.GetFieldData(field)
            
        if self.IsContinuous(field):
          
            # get buckets (< and >) accuracy (most important class)
            loweridx=(where ( d<=threshold ))[0]
            higheridx=(where ( d>threshold ))[0]
            lowerprobs=[]
            higherprobs=[]
            for classval in possibleclassvalues:
                n=(classvalues[loweridx]==classval).sum()
                lowerprobs.append(float(n)/self.nrec)
                n=(classvalues[higheridx]==classval).sum()
                higherprobs.append(float(n)/self.nrec)
            classprob=max(lowerprobs)+max(higherprobs)
        else: # categorical
            classprob=0
            possiblevalues=uniq(d)
            # for all possible values, get maximum class proportion
            for val in possiblevalues:
                idx=self.GetFieldValueIdx(field,val)
                probs=[]
                for classval in possibleclassvalues:
                    n=(classvalues[idx]==classval).sum()
                    probs.append(float(n)/self.nrec)
                classprob+=max(probs)             
          
        return classprob
            
    def GetInfos(self, classfield, showclassfieldstats=False, classification=False, verbose=False):
        
        if showclassfieldstats:
            classvalues=self.GetFieldData(classfield)
            # get list possible class values
            possibleclassvalues=uniq(classvalues)
            print("n classes=",len(possibleclassvalues))
            classdist=Histogram(classvalues,True)
            print("class dist:",classdist)
            
        # probability, entropy (information) histogram and best values associate
        probh={}
        infoh={}
        thresholdih={}
        thresholdph={}
        
        if self.fields.count(classfield)!=1:
            sys.stderr.writelines('unknown class field :'+classfield+'\n')
            print(self.fields)
            return
        
        fields=list(self.fields)    
        fields.remove(classfield)
        
        for field in fields:
            [info,vi,prob,vc]=self.InfoProb(field,classfield,classification)
            infoh[field]=info
            probh[field]=prob
            thresholdih[field]=vi
            thresholdph[field]=vc

        sinfoh=SortHistogram(infoh,False,True)
        
        # print info
        if verbose:
            print("feature entropy sorted:")
            for el in sinfoh:
                countratio=0
                if el[0] in self.uniqvalcountratio:
                    countratio=self.uniqvalcountratio[el[0]]
                print(el)#,countratio*100
            bestfield = sinfoh[0][0]
            print("best feature: %s value: %s entropy: %s classification: %s" %(
                bestfield, thresholdih[bestfield], infoh[bestfield], 
                self.GetClassificationAccuracy(bestfield, classfield, 
                                               thresholdih[bestfield])))           
        
        # print probs
        
        if classification:
            sprobh=SortHistogram(probh,False,True)
            if verbose:
                print("feature probability sorted:")
                for el in sprobh:
                    countratio=0
                    if el[0] in self.uniqvalcountratio:
                        countratio=self.uniqvalcountratio[el[0]]
                    print(el)#,countratio*100
                bestfield=sprobh[0][0]
                print("best feature: %s value: %s classification: %s" %(   
                    bestfield, thresholdph[bestfield],probh[bestfield]))         
        
        return sinfoh, sprobh
       
    def KeepValues(self,field,value):
        try:
            idx=int(field)-1
            field=self.fields[idx]
            print("i=",field)
        except:
            pass
        
        d=self.GetFieldData(field)
        
        if self.IsContinuous(field):
            where(d==float(value))
            return self.data[:,where(d==float(value))]
        else:
            where(d==hash(value))
            return self.data[:,where(d==hash(value))]

    def DistDiff(self,field,xbuckets):
        data=self.GetFieldData(field)
        step=int(floor((self.nrec)/xbuckets))
        diff=zeros(xbuckets,"Float")
        start=0
        for i in range(xbuckets-1):
            stop=start+step
            h1=Histogram(data[start:stop],True)
            h2=Histogram(data[stop:stop+step],True)
            #print start,stop,stop+step
            #print h1,h2
            diff[i]=CategoricalDiff(h1,h2)
            #print d
            start=stop
        print(diff)
        
#util function related to dataset
def merge(dataset1,dataset2):
    if dataset1.header!=dataset2.header:
        sys.stderr(sys._getframe().f_code.co_name+": can't merge those 2 datasets !")
        return None
    nrec=dataset1.nrec+dataset2.nrec
    data=zeros((nrec,self.nfields))
    for i in range(dataset1.s):
        data[1:dataset1.nrec,i]=dataset1.data[:,i]
        data[dataset1.nrec:nrec,i]=dataset2.data[:,i]
        

if __name__ == "__main__":
    ds=Dataset(sys.argv[1])
