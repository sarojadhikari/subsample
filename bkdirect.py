"""
get the bispectrum by direct sampling from a 3d density field distribution
"""
import numpy as np

class densityfield(object):
    """
    class to store and study a 3d density field
    mainly power spectrum and bispectrum implemented
    """
    def __init__(self, dfield, dk=1, skip=1, Lbox=600., subtractmean=True):
        
        self.dmean=np.mean(dfield)
        if (subtractmean):
            self.dfield=dfield-self.dmean # 3d density field as a numpy array
        else:
            self.dfield=dfield
            
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
        self.delninr=None
        
        self.compute_parameters()
    
    def compute_parameters(self):
        """
        compute useful paramters, call this function after changing some of the
        parameters
        """
        self.psfactor = np.power(self.Lbox, 3.0)/np.power(self.ngrid, 6.0)
        self.nkindx=self.nkmax/self.skip
        self.bsfactor = np.power(self.Lbox, 6.0)/np.power(self.ngrid, 9.0)
    
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
        self.kdfield=np.fft.fftn(self.dfield)
        self.freq123=np.fft.fftfreq(self.ngrid, 1.0/self.ngrid)

        return self.kdfield  


    def kvector(self, i, j, l):
        """
        return the kvector for a position in a 3D grid
        """
        return np.array([self.dk*self.freq12[i],
                         self.dk*self.freq12[j],
                         self.dk*self.freq3[l]])

    def get_dki_nr(self):
        """
        generate d_ni(nrv) for bispectrum estimator II
        """
        if self.kdfield==None:
            self.get_dk()
            
        delm = np.array([np.zeros(self.kdfield.shape)]*self.nkindx, dtype=np.cfloat)
        ntra = np.array([np.zeros(self.kdfield.shape)]*self.nkindx, dtype=np.cfloat)
        
        for na in range(self.ngmin, self.ngmax):
            for nb in range(self.ngmin, self.ngmax):
                for nc in range(self.ngmin, self.ngmax):
                    nabs=np.sqrt(na*na+nb*nb+nc*nc)
                    kindx=int(round(nabs/self.skip))
                    if kindx<self.nkindx:
                        m1, m2, m3 = self.kindex(na, nb, nc)
                        delm[kindx][m1][m2][m3]=self.kdfield[m1][m2][m3]
                        ntra[kindx][m1][m2][m3]=1.0
        
        # now FT each of these
        self.delninr=[]
        for ni in range(self.nkindx):
            self.delninr.append(np.fft.fftn(delm[ni]))
            
        # save delninr
        #np.savetxt()
        # get the # of triangles
        self.neqtr2=[]
        self.ntrfdata=[]
        for ni in range(self.nkindx):
            self.ntrfdata.append(np.fft.fftn(ntra[ni]))
        
        self.Bequil=[]  # this lists for triangles in equlateral configuration as functions of k
        self.Bk1k2=[]   # this lists for triangle with k1=10kmin k2=2k1 as a function of angle
        # for each angle we basically need to compute k3    
            
        for ki in range(self.nkindx):
            ntrtemp=0.
            beqtemp=0.
            
            beqtemp2=0.
            
            for nr1 in range(self.ngrid):
                for nr2 in range(self.ngrid):
                    for nr3 in range(self.ngrid):
                        ntrtemp=ntrtemp+np.power(np.real(self.ntrfdata[ki][nr1][nr2][nr3]), 3.0)
                        beqtemp=beqtemp+np.power(np.real(self.delninr[ki][nr1][nr2][nr3]), 3.0)
                        
                        beqtemp2=beqtemp2 + np.real(self.delninr[ki][nr1][nr2][nr3])*np.real(self.delninr[10][nr1][nr2][nr3])*np.real(self.delninr[20][nr1][nr2][nr3])
                        
            print ntrtemp/self.ngrid**3.0
            self.neqtr2.append(int(round(ntrtemp/self.ngrid**3.0)))
            self.Bequil.append(beqtemp)
            self.Bk1k2.append(beqtemp2)
        
        # also generate bispectrum data in more like a squeezed configuration
        # we will take k1=10*kmin, k2=2*k1, and a set of 20 angules between k1 and k2 to define the triangle
                        
            
    def equil2(self):
        if self.delninr==None:
            self.get_dki_nr()
        if self.powerspectrum==None:
            self.compute_powerspectrum()
            
        self.Qequil=np.array([self.Bequil[i]*self.bsfactor/np.power(self.ngrid, 3.0)/self.neqtr2[i]/self.powerspectrum[i]**2.0/3.0 for i in range(len(self.Bequil))])
        
        self.Qequil2=np.array([self.Bk1k2[i]*self.bsfactor/np.power(self.ngrid, 3.0)/self.neqtr2[i]/(self.powerspectrum[10]*self.powerspectrum[20]+self.powerspectrum[10]*self.powerspectrum[i]+self.powerspectrum[20]*self.powerspectrum[i])/3.0 for i in range(len(self.Bequil))])

        
        #for k in range(self.nkindx):
            

    def kindex(self, k1, k2, k3):
        """
        inverse of kvector i.e. given a kvector, provide the app indexes
        """
        return np.array([k1 if k1>=0 else k1+self.ngrid,
                        k2 if k2>=0 else k2+self.ngrid,
                        k3 if k3>=0 else k3+self.ngrid])/self.dk # don't forget to use np.conjugate for negative k3
        #return np.array(np.where(self.freq12==k1)[0][0], np.where(self.freq12==k2)[0][0], np.where(self.freq3==np.abs(k3))[0][0]) # for older numpy version
        #return np.array([self.freq12.index(k1), self.freq12.index(k2), self.freq3.index(np.abs(k3))])
    
    def get_pksample(self):
        """
        generate conj(dk)*dk array for later use
        """
        self.pksample=np.abs(np.conjugate(self.kdfield)*self.kdfield)

    def get_dk_123(self, k1, k2, k3):
        i, j, l = self.kindex(k1, k2, k3)
        return self.kdfield[i][j][l]

    def equil_bispectrum(self, rfac=4):
        """
        a version of the bk_directsample  to obtain equilateral bispectrum
        B(k, k, k) for some set of averaged k=kindex
        """
        if self.kdfield==None:
            self.get_dk()
        
        bk=np.zeros(self.nkindx/rfac, dtype=np.cfloat)
        ntr=np.zeros(self.nkindx/rfac)
        
        pk=np.zeros(self.nkindx/rfac)
        ck=np.zeros(self.nkindx/rfac)
        
        for k11 in range(self.ngmin, self.ngmax):
            for k12 in range(self.ngmin, self.ngmax):
                for k13 in range(self.ngmin, self.ngmax):
                    k1abs=np.sqrt(k11*k11+k12*k12+k13*k13)
                    k1indx=int(round(k1abs/self.skip))
                    
                    if (k1indx < self.nkindx/rfac):
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
                                                ntr[k1indx]=ntr[k1indx]+1.
                                            except:
                                                print k1indx
                                                return 1
        
        for i in range(self.nkindx/rfac):
            if ntr[i]==0:
                ntr[i]=1
            if ck[i]==0:
                ck[i]=1
        
        self.powerspectrum=np.array([self.psfactor*pk[i]/ck[i] for i in range(self.nkindx/rfac)])
        self.paircount=ck

        self.bkdata=bk        
        
        self.eqbispectrum=np.array([self.bsfactor*np.real(bk[i])/ntr[i]/np.power(self.powerspectrum[i],2.0)/3.0 for i in range(self.nkindx/rfac)])  # reduced bispectrum Q(k, k, k)
        self.eqtriangles=ntr
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

    def bispectrum_tests(self):
        """
        make a test case: check I vs II estimators
        """
        
        # generate a 3d 20^3 box with random numbers
