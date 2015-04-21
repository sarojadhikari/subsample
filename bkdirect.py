"""
get the bispectrum by direct sampling from a 3d density field distribution
"""
import numpy as np

class densityfield(object):
    """
    class to store and study a 3d density field
    mainly power spectrum and bispectrum implemented
    """
    def __init__(self, dfield, dk=1, skip=1, Lbox=600.):
        self.dfield=dfield # 3d density field as a numpy array
        self.powerspectrum=None
        self.ngrid=len(dfield)
        self.ngmax=self.ngrid/2
        self.ngmin=-self.ngrid/2
        self.dk=dk
        self.kmax=self.ngrid*self.dk
        self.skip=skip
        self.nkmax=self.ngmax
        self.Lbox=Lbox
        self.kdfield=None
        self.pksample=None
        
        self.compute_parameters()
    
    def compute_parameters(self):
        """
        compute useful paramters, call this function after changing some of the
        parameters
        """
        self.psfactor = np.power(self.Lbox, 3.0)/np.power(self.ngrid, 6.0)
        self.nkindx=self.nkmax/self.skip
        self.bsfactor = self.psfactor/np.power(self.ngrid, 3.0)
    
    def compute_powerspectrum(self):
        """
        compute the power spectrum of the density field
        """
        self.bk_directsample(psonly=True)
        
    def get_dk(self):
        """
        generate the d(k) i.e. FT of density field given a 3D distribution 
        in a cube
        """
        self.kdfield=np.fft.rfftn(self.dfield)
        self.freq12=np.fft.fftfreq(self.ngrid, 1.0/self.ngrid)
        try:
            self.freq3=np.fft.rfftfreq(self.ngrid, 1.0/self.ngrid)
        except:
            self.freq3=np.arange(0, self.ngrid/2+1)

        return self.kdfield  


    def kvector(self, i, j, l):
        """
        return the kvector for a position in a 3D grid
        """
        return np.array([self.dk*self.freq12[i],
                         self.dk*self.freq12[j],
                         self.dk*self.freq3[l]])

    def kindex(self, k1, k2, k3):
        """
        inverse of kvector i.e. given a kvector, provide the app indexes
        """
        return np.array([k1 if k1>=0 else k1+self.ngrid,
                        k2 if k2>=0 else k2+self.ngrid,
                        k3 if k3>=0 else -k3])/self.dk # don't forget to use np.conjugate for negative k3
        #return np.array(np.where(self.freq12==k1)[0][0], np.where(self.freq12==k2)[0][0], np.where(self.freq3==np.abs(k3))[0][0]) # for older numpy version
        #return np.array([self.freq12.index(k1), self.freq12.index(k2), self.freq3.index(np.abs(k3))])
    
    def get_pksample(self):
        """
        generate conj(dk)*dk array for later use
        """
        self.pksample=np.abs(np.conjugate(self.kdfield)*self.kdfield)

    def get_dk_123(self, k1, k2, k3):
        i, j, l = self.kindex(k1, k2, k3)
        try:
            if (k3>=0):
                return self.kdfield[i][j][l]
            else:
                return np.conjugate(self.kdfield[i][j][l])
        except:
            print 123, i, j, l
            return 0

    def equil_bispectrum(self):
        """
        a version of the bk_directsample  to obtain equilateral bispectrum
        B(k, k, k) for some set of averaged k=kindex
        """
        if self.kdfield==None:
            self.get_dk()
        
        bk=np.zeros(self.nkindx, dtype=np.cfloat)
        ntr=np.zeros(self.nkindx)
        
        pk=np.zeros(self.nkindx)
        ck=np.zeros(self.nkindx)
        rfac=10
        for k11 in range(self.ngmin/rfac, self.ngmax/rfac):
            for k12 in range(self.ngmin/rfac, self.ngmax/rfac):
                for k13 in range(self.ngmin/rfac, self.ngmax/rfac):
                    k1abs=np.sqrt(k11*k11+k12*k12+k13*k13)
                    k1indx=int(round(k1abs/self.skip))
                    
                    if (k1indx < self.nkindx):
                        dk1=self.get_dk_123(k11, k12, k13)
                        pk[k1indx]=pk[k1indx]+np.vdot(dk1, dk1)
                        ck[k1indx]=ck[k1indx]+1                        
                    
                        k2max=self.skip*k1indx+self.skip/2
                        
                        for k21 in range(-k2max, k2max):
                            for k22 in range(-k2max, k2max):
                                for k23 in range(-k2max, k2max):
                                    k2abs=np.sqrt(k21*k21+k22*k22+k23*k23)
				    k2indx=int(round(k2abs/self.skip))                                    
				    if (k2indx==k1indx):
                                        # triangle condition
                                        k31=-(k11+k21); k32=-(k12+k22); k33=-(k13+k23)
                                        k3abs=np.sqrt(k31*k31+k32*k32+k33*k33)
                                        k3indx=int(round(k3abs/self.skip))
                                        
                                        if (k3indx==k2indx):
                                            #dk1=self.get_dk_123(k11, k12, k13)
                                            dk2=self.get_dk_123(k21, k22, k23)
                                            dk3=self.get_dk_123(k31, k32, k33)
                                            
                                            try:
                                                bk[k1indx]=bk[k1indx]+(dk1*dk2*dk3)
                                                ntr[k1indx]=ntr[k1indx]+1
                                            except:
                                                print k1indx
                                                return 1
        
        for i in range(self.nkindx):
            if ntr[i]==0:
                ntr[i]=1
            if ck[i]==0:
                ck[i]=1
        
        self.eqbispectrum=np.array([self.bsfactor*bk[i]/ntr[i] for i in range(self.nkindx)])
        self.eqtriangles=ntr
        
        self.powerspectrum=np.array([self.psfactor*pk[i]/ck[i] for i in range(self.nkindx)])
        self.paircount=ck
        return 0

    def bk_directsample(self, psonly=False):
        """
        bispectrum through direct sampling
        """
        if self.kdfield==None:
            self.get_dk()
        if self.pksample==None:
            self.get_pksample()

        bk=np.zeros((self.nkindx, self.nkindx, self.nkindx))
        ntr=np.zeros((self.nkindx, self.nkindx, self.nkindx))     
        
        pk=np.zeros(self.nkindx)
        ck=np.zeros(self.nkindx)
        
        for k11 in range(self.ngmin, self.ngmax):
            for k12 in range(self.ngmin, self.ngmax):
                for k13 in range(self.ngmin, self.ngmax):
                    k1abs=np.sqrt(k11*k11+k12*k12+k13*k13)
                    k1indx=int(round(k1abs/self.skip))
                    i,j,l = self.kindex(k11, k12, k13)
                    if (psonly):
                        if (k1indx < self.nkindx and k1indx >= 0 and k13>=0):
                            dpk=self.pksample[i][j][l]
                            pk[k1indx]=pk[k1indx]+dpk
                            ck[k1indx]=ck[k1indx]+1                    
                    # now proceed to the loop for bispectrum
                    if not(psonly) and k1indx<self.nkindx and k1indx>=0:
                        k2max=self.skip*k1indx+self.skip/2
                        if (k2max>self.ngmax-1):
                            k2max=self.ngmax-1
                        for k21 in range(-k2max, k2max):
                            for k22 in range(-k2max, k2max):
                                for k23 in range(-k2max, k2max):
                                    k2abs=np.sqrt(k21*k21+k22*k22+k23*k23)
                                    k2indx=int(round(k2abs/self.skip))
                                    
                                    if (k2indx <= k1indx and k2indx >= 0):
                                        # now calculate k3=-(k1+k2)
                                        k31=-(k11+k21); k32=-(k12+k22); k33=-(k13+k23)
                                        k3abs=np.sqrt(k31*k31+k32*k32+k33*k33)
                                        k3indx=int(round(k3abs/self.skip))
                                        # check for triangle condition
                                        if (k3indx <= k2indx and k3indx>0 and (k1indx <= k2indx+k3indx)):
                                            dk1=self.get_dk_123(k11, k12, k13)
                                            dk2=self.get_dk_123(k21, k22, k23)
                                            dk3=self.get_dk_123(k31, k32, k33)
                                            try:
                                                bk[k1indx][k2indx][k3indx]=bk[k1indx][k2indx][k3indx]+(dk1*dk2*dk3)
                                                ntr[k1indx][k2indx][k3indx]=ntr[k1indx][k2indx][k3indx]+1
                                            except:
                                                print k1indx, k2indx, k3indx
                                                return 0
                                                                                    
        for i in range(self.nkindx):
            if (ck[i]>0.0):
                pk[i]=self.psfactor*pk[i]/ck[i]

        if (psonly):
            self.powerspectrum=pk
            self.paircount=ck
        else:
            self.bispectrum=bk
            self.ntriangles=ntr
        
        return 0
