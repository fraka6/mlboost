#!/usr/bin/python

# Copyright (C) 2009 Francis Pieraut

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


''' util image function '''

import os
import platform
from math import sqrt
from numpy import reshape, array
#from pylab import imshow, title, subplot, gray, figure,gcf, show as pyshow
#from mlboost.core.pylabhisto import *
from mlboost.core import ppdataset
from mlboost.core import pphisto

#gray()

import time
TSLEEP_LOCK = 2
TSLEEP_CAPTURE = .1
VIDEO_CAPTURE = False

camera = None

if VIDEO_CAPTURE:
    try:
        from VideoCapture import Device
        camera = Device()
    except:
        print("Can't use videoCapture")

    try:
        import opencv
        from opencv import highgui 
        camera = highgui.cvCreateCameraCapture(0)
    except:
        print("can't use opencv")
 
   
def get_image(new_size=None, mode="L"):
    ''' inspired by http://www.jperla.com/blog/2007/09/26/capturing-frames-from-a-webcam-on-linux '''
    ipl_im = highgui.cvQueryFrame(camera)
    # Add the line below if you need it (Ubuntu 8.04+)
    #ipl_im = opencv.cvGetMat(ipl_im)
    #convert Ipl image to PIL image
    im = opencv.adaptors.Ipl2PIL(ipl_im)
    im = im.convert(mode)
        
    if new_size: 
        return array(im.resize(new_size))
    return array(im) 
    
def capture(new_size=None, mode="L"):
    if camera:
        im = camera.getImage().convert(mode)
        
        if new_size: 
            return array(im.resize(new_size))
        return array(im)
    return None

def show_capture(new_size=None):
    im = imshow(capture(new_size))
    pyshow()

im_fname = "image.txt"
file = "/media/share/%s" % im_fname
if platform.system()=="Windows":
    file = "C:/share/%s" % im_fname
lockfile = file + ".lock"
    
def save_capture(new_size=None, fname=file):
    if not os.path.exists(lockfile) and not os.path.exists(fname):
        lf = open(lockfile,'w')
        im = capture(new_size);
        f = open(fname,'wb')
        pickle.dump(im,f)
        f.close()
        lf.close()
        os.remove(lockfile)
        return im
    else:
        time.sleep(TSLEEP_LOCK)
        return save_capture(new_size, fname)
        
def load_capture(fname=file,do_show=True):
    ''' load capture file '''
    if not os.path.exists(lockfile) and os.path.exists(fname):
        lf = open(lockfile,'w')
        f = open(fname,'rb')
        im = pickle.load(f)
        f.close()
        lf.close()
        os.remove(file)
        os.remove(lockfile)
        if do_show:
            imshow(im)
            pyshow()
        return im
    else:
        time.sleep(TSLEEP_LOCK)
        return load_capture(fname, do_show)

def reset():
    ''' force remove of lockfile '''
    if os.path.exists(lockfile):
        os.remove(lockfile)
            
def grab(new_size=(8,8), do_show=False):
    n=0
    while(True):
        im = save_capture(new_size)
        n+=1
        print("grabbing image #%s" %n)
        if do_show:
            imshow(im)
            pyshow()
        time.sleep(TSLEEP_CAPTURE)

def video():
    ''' diplay video grabbed or loaded '''
    n=0
    while(True):
        if camera:
            show_capture()
        else:
            load_capture()
        n+=1
        print("image #%s" %n)
        
def toimage(vector):
    n = int(sqrt(len(vector)))
    return reshape(vector,(n,n))        
             

