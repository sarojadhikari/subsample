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
    dk=1
    #Lb3=np.power(Lbox, 3.0)
    #dx=Lbox/ng
    kmax=ng*dk
    
    ngmax=ng/2+1    
    
    pk=np.zeros(ng)
    ck=np.zeros(ng)
    sk=np.zeros(ng)
    
    dpklist=np.abs(np.conjugate(fdf1)*fdf2)
    
    for i in range(ng):
        for j in range(ng):
            for l in range(int(ngmax)):
                kv=np.array([dk*(i if i<(ngmax) else i-ng),
                             dk*(j if j<(ngmax) else j-ng),
                             dk*l])
                knorm=np.sqrt(np.dot(kv, kv))
                bn=int(np.round(knorm*ng/kmax))
                if (bn>=0 and bn<ng):
                    ck[bn]=ck[bn]+1.0
                    dpk=dpklist[i][j][l] #np.abs(np.vdot(fdf1[i][j][l], fdf2[i][j][l]))
                    pk[bn]=pk[bn]+dpk
                    sk[bn]=sk[bn]+np.power(dpk, 2.0)
    
    klist=np.array([i*dk*2.0*np.pi/Lbox for i in range(ngmax)])
    
    rat = np.power(Lbox, 3.0)/np.power(ng, 6.0)       # the (np.pi)^3 factor comes from numpy.fft convention

    for i in range(ngmax):
        if (ck[i]>0.0):
            pk[i]=rat*pk[i]/ck[i]
            sk[i]=np.power(rat, 2.0)*sk[i]/ck[i]
            sk[i]=np.sqrt((sk[i]-pk[i]*pk[i])/(ck[i]-1.0))

    return np.array([klist, pk[:ngmax], sk[:ngmax], ck[:ngmax]]) 
    
    
def auto_powerspectrum(dfield, Lbox=1.):
    """
    return [k, P(k)] given the density field
    """
    return cross_powerspectrum(dfield, dfield, Lbox)