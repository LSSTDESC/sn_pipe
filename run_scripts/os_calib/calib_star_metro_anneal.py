#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:35:24 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from sn_analysis.sn_os_atmos_calib_tools import load_star_calib,dist
import numpy as np


def init_path(x,y,iorder,ncity):
    """
    function to estimate the path

    Parameters
    ----------
    x : array(float)
        x positions.
    y : array(float)
        y positions.
    iorder : array(int)
        pos order.
    ncity : int
        number of pos.

    Returns
    -------
    path : float
        the path.

    """
    
    path = 0.
    
    for i in range(ncity-1):
        i1 = iorder[i]
        i2 = iorder[i+1]
        path += dist(x[i1],y[i1],x[i2],y[i2])
    #close the loop
    i1 = iorder[ncity-1]
    i2 = iorder[0]
    path += dist(x[i1],y[i1],x[i2],y[i2])    
    
    return path

def revcst(x,y,iorder,n):
    """
    returns the value of the cost function for a proposed path reversal.

    Parameters
    ----------
    x : array(float)
        x positions.
    y : array(float)
        y positions.
    iorder : array(int)
        pos order.
    n : array(int)[2]
        starting and ending cities along 
        the path segment which is to be reversed

    Returns
    -------
    de : float
        cost function value.

    """
    ncity = len(x)
    xx = np.zeros((5,),dtype=float)
    yy = np.zeros((5,),dtype=float)
    """
    n[2] = 1+((n[0]+ncity-2)%ncity) # find the city before n[0]
    n[3] = 1+(n[1]%ncity) #...and the city after n[1]
    """
    #modif to account fo start at 0
    n[2] = ((n[0]+ncity-1)%ncity) # find the city before n[0]
    n[3] = ((n[1]+1)%ncity) #...and the city after n[1]
    
    for j in range(4):
        ii = iorder[n[j]]
        xx[j] = x[ii]
        yy[j] = y[ii]
        
    de = -dist(xx[0],yy[0],xx[2],yy[2])
    de -= dist(xx[1],yy[1],xx[3],yy[3])
    de += dist(xx[0],yy[0],xx[3],yy[3])
    de += dist(xx[1],yy[1],xx[2],yy[2])
    
    return de
    
def reverse(iorder,n):
    """
    path segment reversal. 
    The vector n has as its four elements the first and last positions 
    n[0] and n[1] of the path segment to be reversed and the two positions
    n[2] and n[3] that immediately precede and follow this segment.
    n[2] and n[3] are fond by function revcst.
    
    Parameters
    ----------
    iorder : array(int)
        the present itinerary
    n : array(int)
        segment+positions before and after.

    Returns
    -------
    iorder : array(int)
        the segment from n[0] to n[1] in reversed order

    """
    ncity = len(iorder)
    # this many positions must be swapped to effect the reversal
    nn = (1+((n[1]-n[0]+ncity)%ncity))/2 
    
    nn = int(nn)
    for j in range(nn):
        #start at the ends of the segment and swap pair of cities
        #moving toward the center
        """
        k = 1+((n[0]+j-2)%ncity)
        l = 1+((n[1]-j+ncity)%ncity)
        """
        k = (n[0]+j)%ncity
        l = (n[1]-j+ncity)%ncity
        itmp = iorder[k]
        iorder[k] = iorder[l]
        iorder[l] = itmp
        
    return iorder
    
def trncst(x,y,iorder,n):
    """
    value of the cost function for a proposed path segment transport.
    

    Parameters
    ----------
    x : array(float)
        x positions.
    y : array(float)
        y positions.
    iorder : array(int)
        the present itinerary.
    n : array(int)
        first three elements: starting and ending cities of the path 
        to be transported, and the point among the remaining cities 
        after which it is to be inserted. The actual transport is not 
        performed here.

    Returns
    -------
    de : float
        The cost of the change.

    """
    ncity=len(x)
    xx = np.zeros((7,),dtype=float)
    yy = np.zeros((7,),dtype=float)
    
    """
    n[3] = 1+(n[2]%ncity) #find the city following n[2]
    n[4] = 1+((n[0]+ncity-2)%ncity) #find the city preceding n[1]
    n[5] = 1+(n[1]%ncity) #find the city following n[1]
    """
    n[3] = (n[2]+1)%ncity #find the city following n[2]
    n[4] = (n[0]+ncity-1)%ncity #find the city preceding n[1]
    n[5] = (n[1]+1)%ncity #find the city following n[1]
    
    
    
    for j in range(6):
        ii = iorder[n[j]]
        xx[j] = x[ii]
        yy[j] = y[ii]
        
    de = -dist(xx[1],yy[1],xx[5],yy[5])
    de -= dist(xx[0],yy[0],xx[4],yy[4])
    de -= dist(xx[2],yy[2],xx[3],yy[3])
    de += dist(xx[0],yy[0],xx[2],yy[2])
    de += dist(xx[1],yy[1],xx[3],yy[3])
    de += dist(xx[4],yy[4],xx[5],yy[5])
    
    return de
    
def trnspt(iorder,n):
    """
    The actual transport is done here, once metrop has approved.
    

    Parameters
    ----------
    iorder : array(int)
        input array with the present itinerary
    n : array(int)
        six elements
        - beginning n[0] and end n[1] of the path to be transported
        - adjacent cities n[2] and n[3] between which the path 
        is to be placed
        - cities n[4] and n[5] that precede and follow the path
        - n[3], n[4] and n[5] are calculated by function trncst
        

    Returns
    -------
    iorder : array(int)
        reflect the movement of the path segment - new itinerary

    """
    ncity = len(iorder)
    
    jorder = np.ones((ncity,),dtype=int)
    """
    m1 = 1+((n[1]-n[0]+ncity)%ncity) # find ncities from n[0] to n[1]
    m2 = 1+((n[4]-n[3]+ncity)%ncity) # find ncities from n[3] to n[4]
    m3 = 1+((n[2]-n[5]+ncity)%ncity) # find ncities from n[5] to n[2]
    """
    m1 = (n[1]-n[0]+ncity)%ncity # find ncities from n[0] to n[1]
    m2 = (n[4]-n[3]+ncity)%ncity # find ncities from n[3] to n[4]
    m3 = (n[2]-n[5]+ncity)%ncity # find ncities from n[5] to n[2]
    
    
    nn = 0
    
    for j in range(m1+1):
        jj = (j+n[0])%ncity # copy the chosen fragment
        jorder[nn] = iorder[jj]
        nn += 1
    print(m1,nn)
    for j in range(m2+1):
        jj = (j+n[3])%ncity # copy the segment from n[3] to n[4]
        jorder[nn] = iorder[jj]
        nn += 1
    for j in range(m3+1):
        jj = (j+n[5])%ncity # copy the segment from n[5] to n[2]
        jorder[nn] = iorder[jj]
        nn += 1
    #copy jorder back into order
    for j in range(ncity):
        iorder[j] = jorder[j]
        
    del jorder
    
    return iorder
    
def metrop(de,t):
    """
    Metropolis algorithm

    Parameters
    ----------
    de : float
        cost value.
    t : float
        temperature determined by the annealing schedule.

    Returns
    -------
    bool
        issues a verdict on whether to accept a configuration that leads 
        to a change de in the objective function e.
        if de < 0 metrop=1
        if de > 0 metrop is only true with probability exp(-de/t)

    """
    
    rr = np.random.random_sample()
        
    return (de < 0 or rr < np.exp(-de/t))
    
def get_segment(ncity,n):
    """
    function to grab initial segment

    Parameters
    ----------
    ncity : int
        number of pos.
    n: array(int)
       segment array - empty

    Returns
    -------
    n : array(int)
        segment array.

    """
    
    #n = np.zeros((7,),dtype=int)
    
    nn = 0
    
    while nn < 3:
        n[0] = np.random.randint(0,ncity,1)[0] #beginning of the segment
        n[1] = np.random.randint(0,ncity-1,1)[0] #end of the segment
        if n[1] >= n[0]:
            n[1] += 1
        #nn is the number of cities not on the segment
        nn = 1+((n[0]-n[1]+ncity-1)%ncity)
    
    return n,nn
    

fName = 'input/scheduler/calspec_gaia_matching.csv'

df = load_star_calib(fName)

idx = df['RA_deg'] >= 0. 
idx &= df['RA_deg'] <= 360.

df=df[idx]
df = df.reset_index()
df['iorder'] = df.index

x=df['RA_deg'].to_numpy()
y=df['Dec_deg'].to_numpy()
iorder = df['iorder'].to_numpy()
ncity = len(df)

n = np.zeros((7,),dtype=int)

path = init_path(x,y,iorder,ncity)

print('init path',path)

njt = 100
nover = 100*ncity
nlimit = 10*ncity
tfactr=0.9
t = 0.5

for jt in range(njt):
    nsucc = 0
    for k in range(nover):
        #grab initial segment
        n,nn = get_segment(ncity,n)
        #segment reversal or transport?
        rr = np.random.random_sample()
        if rr > 0.5: #do a transport
            rrb = np.random.random_sample()
            n[2] = n[1]+int(np.abs(nn-1)*rrb+1) 
            n[2] %= ncity #transport the location not on the path
            de = trncst(x,y,iorder,n) # calculate cost
            ans = metrop(de,t) #consult the oracle
            if ans:
                nsucc += 1
                path += de
                trnspt(iorder,n) # carry out the transport
        else: #do a path reversal
            de = revcst(x,y,iorder,n) # estimate cost
            ans = metrop(de,t) # consult the oracle
            if ans:
                nsucc +=1
                path += de
                reverse(iorder,n) #carry out the reversal
        if nsucc >= nlimit:
            break
        print('temp',t)
        print('path length',path)
        print('succesfull moves',nsucc)
        t *= tfactr
        if nsucc == 0:
            break
        



"""
print(df)

print(x)

print(x[3])

x[3] = -2

print(x)

"""

"""
fig, ax = plt.subplots()

print(df.columns)
ax.plot(df['RA_deg'],df['Dec_deg'],'ko')

plt.show()
"""