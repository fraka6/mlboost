#!/usr/local/bin/python

# pputil is part of ppboost
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


import string
import types
import time
from numpy import *

# list of list to array
def list2array(l):
    nx=len(l[0])
    ny=len(l)
    x=zeros((nx,ny),float)
    for j in range(ny):
        x[0,j]=float(l[j][0])
        x[1,j]=float(l[j][1])
    return x

def nlines(filename):
    n=0
    try:
        f=open(filename)
        n=len(f.readlines())
    except Exception as e:
        print(e)
    f.close()
    return n

def header(filename):
    try:
        f=open(filename)
        return f.readline()
    except Exception as e:
        print(e)
        return ""

# x should be an array
def diff(x):
    a=x[0:-1]
    b=x[1:]
    return b-a
    
    return 
# epoch to time convertion (-5hrs -> epoch2time(x,5)----------
def epoch2time(epoch,hrs=0):
    cst=int(hrs)*60*60#time.mktime(time.gmtime(0))
    return time.ctime(epoch-cst)

# get time regex function ex: 2005,2,28,15,0,25 ->epoch 
def gettime1(s):
    #string list
    sl=re.match("(.*),(.*),(.*),(.*),(.*),(.*)",s).groups()
    #convert to int list
    il=[]
    for i in sl:
        il.append(int(i))
    # add 3 value needed by python to bring it to 9 parameters
    for i in range(3):
        il.append(0)
    return time.mktime(il)*1000

def gettime2(s):
    #string list
    try:
        sl=re.match("(.*)/(.*)/(.*) (.*):(.*):([^.]*).*",s).groups()
        #convert to int list
        il=[]
        for i in sl:
            il.append(int(i))
        # add 3 value needed by python to bring it to 9 parameters
        for i in range(3):
            il.append(0)

        return time.mktime(il)
    
    except Exception as e:
        print(e,s)
    
    
# get uniq elements from a list
# Fastest without order preserving
# thanks Raymond Hettinger 
#def uniq(alist)    # Fastest order preserving
#    set = {}
#    return [set.setdefault(e,e) for e in alist if e not in set]
def uniq(alist):    
    set = {}
    list(map(set.__setitem__, alist, []))
    return list(set.keys())

    
def StringifyValues(values,separator=",",endcaracter="\n"):
    answer=""
    try:
        answer= str(values).replace("''","NULL").replace("'","").replace('[',"").replace(']',"")+endcaracter
        if separator!=",":
            answer = answer.replace(",",separator)
    except:
        sys.stderr.writelines("Problem in Stringify")
    return answer
	    
# Save dictionary--------------------------------------
def SaveDict(dict,filename):
	
	myfile = open(filename,'w')
	i=0
	keys=list(dict.keys())
	keys.sort()
	for k in keys:
		t=type(dict[k])
	try:
		
		listvals=[]
		if  t is dict:
			listvals=StringifyValues(list(dict[k].keys()))
		elif t is list:
			listvals=StringifyValues(dict[k])
			size=str(len(dict[k]))
			myfile.writelines(str(i)+","+k+"("+size+"):"+str(listvals))
			i+=1
	except Exception as e:
		print("trouble with ",k,"in save Dictionary",e)#,dict[k]
		
	myfile.close()
	print("Creation of:",filename)

# Load dictionary---------------------------------------
def LoadDictFile(filename,rectype="list"):
	print("Loading dictionary struct from",filename,"...")
	data={}
	i=0
	myfile = open(filename,'r')
	for line in myfile.readlines():
		i+=1
		# parse line
		m=re.match("([0-9]*),(.*)\([0-9]*\):(.*)",line)

		# assign if line is std
		if m==None:
			print("problem reading line ",i," :",line)
		else:
			
			# create hash
			if rectype=="list":
				data[m.group(2)]=m.group(3).split(',')
			elif rectype=="hash":
				#print m.groups()
				data[m.group(2)]={}
				df=data[m.group(2)]
				# insert values
				kvalues=m.group(3)
				m=re.match("{(.*)}",kvalues)
				kvalues=m.group(1).split(',')
				for kv in kvalues:
					try:
						k,v=kv.split(": ") 
						df[k]=float(v)
					except:
						print("Problem with:",str(kv))
	return data

class FileCache:
    '''Caches the contents of a set of files.
    Avoids reading files repeatedly from disk by holding onto the
    contents of each file as a list of strings.
    '''

    def __init__(self):
        self.filecache = {}
        
    def grabFile(self, filename):
        '''Return the contents of a file as a list of strings.
        New line characters are removed.
        '''
        if filename not in self.filecache:
            f = open(filename, "r")
            self.filecache[filename] = string.split(f.read(), '\n')
            f.close()
        return self.filecache[filename]

# thanks to Edward Hartfield
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/473872
def number_format(num, places=0,sep=" "):
   """Format a number with grouped thousands and given decimal places"""

   places = max(0,places)
   tmp = "%.*f" % (places, num)
   point = tmp.find(".")
   integer = (point == -1) and tmp or tmp[:point]
   decimal = (point != -1) and tmp[point:] or ""

   count =  0
   formatted = []
   for i in range(len(integer), 0, -1):
       count += 1
       formatted.append(integer[i - 1])
       if count % 3 == 0 and i - 1:
           formatted.append(sep)

   integer = "".join(formatted[::-1])
   return integer+decimal    
