# master script to analyse a large volume density field values by breaking down into smaller subvolumes
# makes use of ssutils.py
# can work on local non-Gaussianity for each subvolume, as one can just add fNL(d^2-<d^2>_large) term in each subvolume

import numpy as np
from ssutils import read_delta_map
#from powerspectrum import auto_powerspectrum
from bkdirect import densityfield

class subsample(object):
    """
    class to define and manipulate (i.e. generate useful subsamples) a set of files 
    (with density fields on a grid)  N files each with Ngrid x Ngrid x Ngrid/N 
    density values generated from N cores using Roman's 2lpt code, binary collection
    of float (8 bytes each)
    """
    
    def __init__(self, filebase="pot.delta", Nfiles=32, Lmesh=2048, subx=4, NSIDE=256):
        self.filebase = filebase
        self.Nfiles = Nfiles
        self.Lmesh = Lmesh
        self.Xmesh = Lmesh/Nfiles
        self.subx = subx
        self.subgrid = Lmesh/subx
        self.segments = self.subgrid/self.Xmesh
        self.Nsubs=int(np.power(self.subx, 3.0))
        self.nside=NSIDE
        
    def set_basedir(self, basedir):
        self.basedir=basedir
    
    def set_outputdir(self, odir):
        self.outputdir=odir
    
    def temp_fname(self, sx, sy, tc, fbase="segment"):
        return "temp/"+fbase+"_"+str(sx)+"_"+str(sy)+"_"+str(tc)+".npy"
        
    def read_density_file(self, fileN):
        """
        read the binary map number #mapN and return its data as a numpy array
        """
        return read_delta_map(self.basedir+self.filebase+"."+str(fileN),  Lmesh=self.Lmesh, Xmesh=self.Xmesh)
        
    def GenerateSegments(self, fNi):
        """
        given a file with Ngrid x Ngrid x Ngrid/N density values, generate Seg segments
        """
        print fNi
        width=self.Lmesh/self.subx
        
        data=self.read_density_file(fNi)
        for sx in range(self.subx):
            for sy in range(self.subx):
                data4=data[:, sx*width:(sx+1)*width, sy*width:(sy+1)*width]
            # save these
                np.save(self.outputdir+self.temp_fname(sx, sy, fNi), data4)
    
    def GenerateRandomSubSample(self, seed):
        """
        randomly generate a subsample of the size specified in self.subx
        first determine the location of the random box and then get the relevant data from
        the relevant segments to get the box
        """
        return 0
            
    def GenerateSubSample(self, seg, sx, sy, ps=False, Lsub=1.):
        """
        generate a subsample by combining relevant segements
        """
        si=seg*self.segments
        subN=seg*self.subx*self.subx+sx*self.subx+sy
        
        try:
            data=np.load(self.outputdir+self.temp_fname(sx, sy, si))
            for tc in range(1, self.segments):
                data=np.append(data, np.load(self.outputdir+self.temp_fname(sx, sy, si+tc)), axis=0)
        except:
            print "error loading files, perhaps the segments are not fully generated\n"
            return 0
        
        np.save(self.outputdir+"stat_"+str(subN)+".npy", np.array([np.mean(data), np.var(data)]))
        # save this subsample and also generate and save 2D projections
        np.save(self.outputdir+"dmap_"+str(subN)+".npy", data)
        
        if (ps):
            df=densityfield(data, skip=1, Lbox=Lsub)
            df.equil2()
            if (subN<self.Nsubs/10):
                np.save(self.outputdir+"dkinr_"+str(subN)+".npy", df.delninr)
            np.save(self.outputdir+"pslists_"+str(subN)+".npy", np.array([df.powerspectrum, df.paircount]))
            np.save(self.outputdir+"eqbis_"+str(subN)+".npy", np.array([df.Qequil, df.neqtr2]))
      
        #hmap=cube_to_healpix(data, self.nside)
        #np.save(self.outputdir+"hmap_"+str(subN)+".npy", hmap)
        print subN

class subsubsample(object):
    """
    reads a subsample (or more generally a cubic box with density values)
    and subdivide into subsubsamples (cubic) and generate 
    power spectrum and other useful statistics
    """              
    def __init__(self, filebase="dmap", sn=0, Lmesh=200, subx=2):
        self.filebase = filebase
        self.samplenumber = sn
        self.Lmesh = Lmesh
        self.subx = subx
        self.subgrid = Lmesh/subx
        self.Nsubs=int(np.power(self.subx, 3.0))
        
    def set_basedir(self, basedir):
        self.basedir=basedir
    
    def set_outputdir(self, odir):
        self.outputdir=odir
        
    def read_density_file(self):
        """
        read the numpy map number #mapN and return its data as a numpy array
        """
        return np.load(self.basedir+self.filebase+"_"+str(self.samplenumber)+".npy")
    
    def GenerateSSSamples(self, ps=True, Lsub=1.):
        try:
            data=self.read_density_file()
        except:
            print "error loading density file\n"
            return 0
        
        for sx in range(self.subx):
            for sy in range(self.subx):
                for sz in range(self.subx):
                    width=self.Lmesh/self.subx
                    ssdata=data[sx*width:(sx+1)*width, sy*width:(sy+1)*width, sz*width:(sz+1)*width]
                    # save this and also generate power spectrum stats etc
                    subN=(sx*self.subx*self.subx+sy*self.subx+sz)+self.samplenumber*self.subx*self.subx*self.subx                    
                    print subN
                    np.save(self.outputdir+"dmap_"+str(subN)+".npy", ssdata)
                    np.save(self.outputdir+"stat_"+str(subN)+".npy", np.array([np.mean(ssdata), np.var(ssdata)]))
                    
                    if (ps):
                        df=densityfield(ssdata, skip=1, Lbox=Lsub)
                        df.equil2()
                        if (subN<self.Nsubs/8):
                            np.save(self.outputdir+"dkinr_"+str(subN)+".npy", df.delninr)
                            
                        np.save(self.outputdir+"pslists_"+str(subN)+".npy", np.array([df.powerspectrum, df.paircount]))
                        np.save(self.outputdir+"eqbis_"+str(subN)+".npy", np.array([df.Qequil, df.neqtr2]))
        