# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:29:08 2015

    various useful tools for subsampling analysis

@author: adhikari
"""

import numpy as np
import healpy as hp
import struct

def read_delta_map(deltamapbinary, Lmesh=2048, Xmesh=64):
    fl = open(deltamapbinary, "rb")
    data=np.array([np.array([np.array([0.]*Lmesh)]*Lmesh)]*Xmesh) 
    fl.seek(0)
    for i in range(Xmesh):
        for j in range(Lmesh):
            for k in range(Lmesh):
                btes=fl.read(8)
                data[i][j][k]=struct.unpack('d', btes)[0]                
    fl.close()
    return data

def distance(p1, p2):
    return np.sqrt(np.power(p1[0]-p2[0], 2.0)+np.power(p1[1]-p2[1], 2.0)+np.power(p1[2]-p2[2], 2.0))

def vector(pos, center=[0.5, 0.5, 0.5], dist=0):
    if (dist==0):
        dist=distance(pos, center)
    vec = np.array([pos[0]-center[0], pos[1]-center[0], pos[2]-center[0]])
    return vec/dist
    
def cube_to_healpix(data, NSIDE=16, lowcount=10):
    """
    * identify the angle of each grid point from the center of the grid
    * if this grid point lies inside the radius, then project to the relevant healpix pixel
    * return a mean and variance map
    """
    nbins=len(data)
    ccord=(nbins-1)/2.0   # count starts from 0

    mmap=np.array([0.]*hp.nside2npix(NSIDE))
    vmap=np.array([0.]*hp.nside2npix(NSIDE))
    pmap=np.array([0.]*hp.nside2npix(NSIDE))

    # now loop over the 3d grid points
    for nx in range(nbins):
        for ny in range(nbins):
            for nz in range(nbins):
                dis=abs(distance(np.array([nx, ny, nz]), np.array([ccord, ccord, ccord])))
                if (dis<ccord/2.0 and dis>ccord/8.0): 
                    # the grid point is inside
                    vx,vy,vz=vector([nx,ny,nz], [ccord,ccord,ccord])
                    pix=hp.vec2pix(NSIDE, vx, vy, vz)
                    pmap[pix]=pmap[pix]+1.0
                    d=data[nx][ny][nz]
                    mmap[pix]=mmap[pix]+d
                    vmap[pix]=vmap[pix]+np.power(d,2.0)
    
    # also generate mask for low statistic points for which count is less than 10    
    mask=np.logical_not(pmap>lowcount)
    return [pmap, mmap, vmap, mask]           

def get_hem_Cls(skymap, direction, LMAX=256):
    """
    from the given healpix skymap, return Cls for two hemispheres defined by the
    direction given, useful to study the possible scale dependence of power modulation
    
    direction should be a unit vector
    """
    # generate hemispherical mask
    NPIX=len(skymap)
    NSIDE=hp.npix2nside(NPIX)
    maskp=np.array([0.]*NPIX)
    #maskm=np.array([1.]*NPIX)
    disc=hp.query_disc(nside=NSIDE, vec=direction, radius=0.0174532925*90.)
    maskp[disc]=1.
    #maskm[disc]=0.
    map1=hp.ma(skymap)
    map1.mask=maskp
    Clsp=hp.anafast(map1, lmax=LMAX)
    map1.mask=np.logical_not(maskp)
    Clsm=hp.anafast(map1, lmax=LMAX)

    return [Clsp, Clsm]    
 
