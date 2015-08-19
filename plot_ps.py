"""
python class to read and plot the stored subsampled power spectra
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'ytick.major.pad': 9})
matplotlib.rcParams.update({'xtick.major.pad': 7})

class SSPowerSpectra(object):
    
    def __init__(self, basedir="/gpfs/home/sza5154/scratch", name="temp", seed="2321", Nsubs=64, Lsub=600.):
        self.basedir=basedir
        self.name=name
        self.seed=seed
        self.Nsubs=Nsubs
        
        if (Nsubs<=64):
            self.datadir=basedir+"/"+name+"/"+seed+"/pot.prim64/"
        else:
            self.datadir=basedir+"/"+name+"/"+seed+"/pot.prim512/"
            
        self.filebase="pslists"
        self.fbbispec="eqbis"
        self.plt=plt
        self.pfactor=1.     # just multiply each power spectra by this factor
        self.Lsub=Lsub
        self.powerspectrum=None
        self.weightedpower=None
        self.normalized=False

    def iBk(self):
        """
        get the integrated bispectrum (squeezed) by correlating P(k, sub) with
        the overdensity
        """
        if self.powerspectrum==None:
            self.average_ps()
            
        self.iBkmean=np.array([])
        self.iBksigma=np.array([])
        self.iBkdata=[]
        sigmasqL=np.var(self.ds)
        # NEED WEIGHTED AVERAGE?
        for i in range(0, len(self.klist)):
            iBkdat=np.array([self.powerspectra[j][i]*self.ds[j]/self.powerspectrum[i]/sigmasqL for j in range(self.Nsubs)])
            self.iBkmean=np.append(self.iBkmean, np.mean(iBkdat))
            self.iBksigma=np.append(self.iBksigma, np.sqrt(np.var(iBkdat)))
            self.iBkdata.append(iBkdat)
            
        # get <P(k) \bar{phi}> correlation per subvolume -- this does not weigh the higher k more.
        self.iBksubs = np.mean(self.iBkdata, axis=0)

    def weighted_ps(self, mfactor=1.1):
        """
        generate weighted ps i.e. for each power spectrum compute a average 
        weighted by k
        """
        self.weightedpower=[]
        #ksum=np.sum(self.psdata[self.klist)
        Nk=int(len(self.klist)/mfactor)
        for i in range(self.Nsubs):
            nsum=np.sum(self.psdata[i][1][0:Nk])
            total=np.sum(np.array([self.psdata[i][1][j]*self.powerspectra[i][j] for j in range(Nk)]))
            self.weightedpower.append(total/nsum)
            
        # also find correlation
        self.corr=[]
        for i in range(self.Nsubs):
            self.corr.append(self.ds[i]*self.weightedpower[i])
        
        self.corr_mean=np.mean(self.corr)
        self.corr_sigma=np.sqrt(np.var(self.corr))
             
    def average_ps(self):
        """
        get the average power spectrum from the subsamples
        """
        
        self.powerspectrum=np.average(self.powerspectra, axis=0)
    
    def normalize(self, reverse=False):
        """
        divide individual power spectra by the average
        """
        if self.powerspectrum==None:
            self.average_ps()
        for sub in range(self.Nsubs):
            if not(reverse):
                self.powerspectra[sub]=self.powerspectra[sub]/self.powerspectrum
                self.normalized=True
            else:
                self.powerspectra[sub]=self.powerspectra[sub]*self.powerspectrum
                self.normalized=False
           
    def read_statistics(self):
        """
        read the power spectrum files and store appropriate data in numpy
        arrays
        """
        self.psdata=[]
        self.powerspectra=[]
        self.ds=[]
        self.dsigma=[]
        self.bsdata=[]
        self.eqbispectra=[]
        self.fNLeq=[]
        
        for sub in range(self.Nsubs):
            self.psdata.append(np.load(self.datadir+self.filebase+"_"+str(sub)+".npy"))
            self.powerspectra.append(np.trim_zeros(self.psdata[-1][0][1:]))
            self.bsdata.append(np.load(self.datadir+self.fbbispec+"_"+str(sub)+".npy"))
            self.eqbispectra.append(self.bsdata[-1][0][1:len(self.powerspectra[-1])])
            
            self.ds.append(np.load(self.datadir+"stat_"+str(sub)+".npy")[0])
            self.dsigma.append(np.sqrt(np.load(self.datadir+"stat_"+str(sub)+".npy")[1]))

        self.klist=np.arange(1, len(self.powerspectra[-1]))*(2.*np.pi/self.Lsub)
        # subtract the mean ds
        self.ds = self.ds - np.mean(self.ds)
        self.fNLeq=np.mean(self.eqbispectra, axis=0)
        self.fNLeqsubs=np.mean(self.eqbispectra, axis=1)
        
    def plot_ps(self, show=False, density=True, pcolor="r", mcolor="b", lw=1.0):
        """
        plot all subsampled power spectra in a plot
        """
        
        if (density):
            """ also read the local overdensity value and plot line colors according to
            the density value, + = red, - = blue; adjust alpha accordingly
            """
            if len(self.ds)<self.Nsubs:
                print "no density data"
                return 0
            ads=np.abs(self.ds)
	    meands=np.mean(self.ds)
            mads=np.max(ads)
            normds=np.array([ads[i]/mads for i in range(len(ads))])
        self.normds=normds

        for sub in range(self.Nsubs):
            #print sub
            if not(density):
                self.plt.plot(self.klist, self.pfactor*self.powerspectra[sub])
            else:
                if self.ds[sub]>meands:
                    self.plt.plot(self.klist[:-1], self.pfactor*self.powerspectra[sub][1:-1], color=pcolor, alpha=normds[sub], linewidth=lw)
                else:
                    self.plt.plot(self.klist[:-1], self.pfactor*self.powerspectra[sub][1:-1], color=mcolor, alpha=normds[sub], linewidth=lw)
        #self.plt.xlim(self.klist[1], 0.1)
        #if (self.normalized):
        #    self.plt.ylim(0.0,2)
        #else:
        #    self.plt.ylim(500, 50000)
        #    self.plt.yscale('log')
        
        self.plt.xlabel(r"$k {\rm (h/Mpc)}$")
        if (self.normalized):
            self.plt.ylabel(r"$P(k)^{\rm subvolume} / P(k)$")
            self.plt.yscale('linear')
        else:
            self.plt.ylabel(r"${\rm P}(k)\; {\rm (Mpc/h)}^3$")
            self.plt.yscale('log')
        
        if (show):
            self.plt.show()
        
    
    def plot_bs(self, show=False, density=True, pcolor="r", mcolor="b", lw=1.0):
        """
        plot eq bispectra for subsamples color coded by the average density of the subsamples
        """
        if (density):
            """ also read the local overdensity value and plot line colors according to
            the density value, + = red, - = blue; adjust alpha accordingly
            """
            if len(self.ds)<self.Nsubs:
                print "no density data"
                return 0
            ads=np.abs(self.ds)
	    meands=np.mean(self.ds)
            mads=np.max(ads)
            normds=np.array([ads[i]/mads for i in range(len(ads))])
        self.normds=normds

        for sub in range(self.Nsubs):
            #print sub
            if not(density):
                self.plt.plot(self.klist, self.fNLeq[sub])
            else:
                if self.ds[sub]>meands:
                    self.plt.plot(self.klist[1:-1], self.eqbispectra[sub][1:-1], color=pcolor, alpha=normds[sub], linewidth=lw)
                else:
                    self.plt.plot(self.klist[1:-1], self.eqbispectra[sub][1:-1], color=mcolor, alpha=normds[sub], linewidth=lw)
        
        self.plt.xlabel(r"$k {\rm (h/Mpc)}$")
        self.plt.ylabel(r"${\rm Q}(k)$")
        self.plt.xscale('log')
        #self.plt.yscale('log')
        
        if (show):
            self.plt.show()
        
