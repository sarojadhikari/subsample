"""
estimate isotropic power spectrum P(k) from a 3D (cubic) density field
"""

import numpy as np

def ft_CIC(k, dx):
    return np.power(np.sinc(k[0]*dx/2)*np.sinc(k[1]*dx/2)*np.sinc(k[2]*dx/2), 2.0)

def cross_powerspectrum(dfield1, dfield2, Lbox=1.):
    """
    return the cross power spectrum between two 3D density fields
    """
    ng=len(dfield1)
    fdf1=np.fft.rfftn(dfield1)
    fdf2=np.fft.rfftn(dfield2)
    
    dk=2.0*np.pi/Lbox
    dx=Lbox/ng
    dx3=np.power(dx, 3.0)
    ng3=np.power(ng, 3.0)
    Lb3=np.power(Lbox, 3.0)
    kmax=np.pi/dx
    
    pk=np.zeros(ng)
    ck=np.zeros(ng)
    sk=np.zeros(ng)
    
    for i in range(ng):
        for j in range(ng):
            for l in range(int(ng/2+1)):
                kv=np.array([dk*(i if i<(ng/2+1) else i-ng),
                             dk*(j if j<(ng/2+1) else j-ng),
                             dk*l])
                knorm=np.sqrt(np.dot(kv, kv))
                bn=int(np.round(knorm*ng/kmax))
                if (bn>=0 and bn<ng):
                    ck[bn]=ck[bn]+1.0
                    dpk=np.abs(np.vdot(fdf1[i][j][l], fdf2[i][j][l]))/np.power(ft_CIC(kv, dx), 2.0)
                    pk[bn]=pk[bn]+dpk
                    sk[bn]=sk[bn]+np.power(dpk, 2.0)
    
    klist=np.array([dk*i for i in range(ng)])
    
    for i in range(ng):
        if (ck[i]>0.0):
            pk[i]=pk[i]*Lb3*dx3/ng3/ck[i]
            sk[i]=sk[i]*Lb3*np.power(dx3/ng3, 2.0)/ck[i]
            sk[i]=np.sqrt((sk[i]-pk[i]*pk[i])/(ck[i]-1.0))

    return np.array([klist, pk, sk, ck]) 
    
    
def auto_powerspectrum(dfield, Lbox=1.):
    """
    return [k, P(k)] given the density field
    """
    return cross_powerspectrum(dfield, dfield, Lbox)