
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


from numpy import *
from math import log
# binary entropy 
def binaryentropy(p):
    if p>0.01:
        return -(p*log(p,2)+p*log(1-p,2))
    return 0
    
# transform the probability into feature correlation
def probs2corr(probs,counts):
    if counts.sum()==0:
	    return 1
    # apply a simple shift transformation to ensure p=100%->info=100%
    infos=list(map(binaryentropy,probs-.5))
    return (infos*counts/counts.sum()).sum()
