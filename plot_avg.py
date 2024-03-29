"""
make useful scripts to plot averaged quantities, averaged over
    * subvolumes,
    * seeds, or
    * scale (k)

quantities like <P_subvolume delta_L> and <Q_eq,L delta_L>
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'ytick.major.pad': 9})
matplotlib.rcParams.update({'xtick.major.pad': 7})

from plot_ps import SSPowerSpectra

xsize=6.0
ysize=4.5

def figurebox(xs=xsize, ys=ysize):
    plt.figure(num=None, figsize=(xs, ys))
    plt.tick_params(which='both', color="gray")

def PPhideltaSqL(TYPE="fNL", NG=100, seeds=["7222", "4233", "221192", "52210"], subtractG=True, Nmax=512):
    PPhideltaL(TYPE=TYPE, NG=NG, seeds=seeds, subtractG=subtractG, Nmax=Nmax, var=True)

def PPhideltaL(TYPE="fNL", NG=100, seeds=["7222", "4233", "221192", "52210"], subtractG=True, Nmax=512, var=False):
    NAME="normal"+TYPE+str(NG)
    for SEED in seeds:
        ssg=SSPowerSpectra(name="normalfNL0", seed=SEED, Nsubs=Nmax)
        ssg.read_statistics()
        sigmasqL=np.var(ssg.ds)
        ss=SSPowerSpectra(name=NAME, seed=SEED, Nsubs=Nmax)
        ss.read_statistics()
        ss.iBk(sigmasqL, variance=var)

        iBkmean = ss.iBkmean

        if (subtractG):
            ssg.iBk(sigmasqL, variance=var)
            iBkmean = iBkmean - ssg.iBkmean

        plt.plot(ss.klist, iBkmean)

    kmin=2.*np.pi/ss.Lsub
    kmax=kmin*33
    plt.xlim(kmin, kmax)
    plt.xlabel(r'$k \left( {\rm h/Mpc} \right)$')
    plt.ylabel(r'$\langle P_{\rm sv}(k) \bar{\delta}_{\rm sv} \rangle/P(k)/\sigma_{\phi_L}^2$')
    plt.title(r'$f_{\rm NL}='+str(NG)+'$', y=1.02)
    plt.show()

def PQeqdeltaL(TYPE="D3", NG=100000, seeds=["7222", "4233", "221192"], subtractG=True, Nmax=512, factor=1.):
    NAME="normal"+TYPE+str(NG)
    for SEED in seeds:
        ss=SSPowerSpectra(name=NAME, seed=SEED, Nsubs=Nmax)
        ss.read_statistics()
        Qeqdeltamean = np.mean(ss.fNLeqds, axis=0)

        if (subtractG):
            # get the Gaussian piece to subtractG
            ssg=SSPowerSpectra(name="normalfNL0", seed=SEED, Nsubs=Nmax)
            ssg.read_statistics()
            QeqdeltameanG = np.mean(ssg.fNLeqds, axis=0)
            Qeqdeltamean = Qeqdeltamean - QeqdeltameanG

        Nks=len(Qeqdeltamean)
        plt.plot(ss.klist[:Nks], factor*Qeqdeltamean)

    sqLimitTheory = np.array([np.var(ssg.ds)*NG*2 for i in range(Nks)])
    plt.plot(ss.klist[:Nks], sqLimitTheory, color="black", linestyle="dashed", linewidth=1.5, alpha=0.4)
    plt.ylabel(r'$\left\langle Q^{\rm equil}(k) \bar{\delta}_{\mathbf{r}_L} \right\rangle$')
    plt.title(r'$g_{\rm NL}^{\rm T1}='+str(NG)+'$', y=1.02)
    kmin=2.*np.pi/ss.Lsub
    kmax=kmin*33
    plt.xlim(kmin, kmax)

    plt.show()
