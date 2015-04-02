# master script to analyse a large volume density field values by breaking down into smaller subvolumes
# makes use of ssutils.py
# can work on local non-Gaussianity for each subvolume, as one can just add fNL(d^2-<d^2>_large) term in each subvolume

import numpy as np
from ssutils import cube_to_healpix, read_delta_map
from powerspectrum import cross_powerspectrum, auto_powerspectrum

class subsample(object):
    """
    class to define and manipulate (i.e. generate useful subsamples) a set of files 
    (with density fields on a grid)  N files each with Ngrid x Ngrid x Ngrid/N 
    density values generated from N cores using Roman's 2lpt code, binary collection
    of float (8 bytes each)
    """
    
    def __init__(self, filebase="gauspot", Nfiles=32, Lmesh=2048, subx=4, NSIDE=256):
        self.filebase = filebase
        self.Nfiles = Nfiles
        self.Lmesh = Lmesh
        self.Xmesh = Lmesh/Nfiles
        self.subx = subx
        self.subgrid = Lmesh/subx
        self.segments = self.subgrid/self.Xmesh
        self.Nsubs=int(np.power(self.subx, 3.0))
        self.nside=NSIDE
        self.dsq=np.zeros(self.Nsubs)
        
    def set_basedir(self, basedir):
        self.basedir=basedir
    
    def set_outputdir(self, odir):
        self.outputdir=odir
        
    def temp_fname(self, sx, sy, tc):
        return "temp/segment_"+str(sx)+"_"+str(sy)+"_"+str(tc)+".npy"
        
    def read_density_file(self, fileN):
        """
        read the binary map number #mapN and return its data as a numpy array
        """
        return read_delta_map(self.basedir+self.filebase+"."+str(fileN), Lmesh=self.Lmesh, Xmesh=self.Xmesh)
        
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
            
        self.dsq[subN]=np.var(data)
        # save this subsample and also generate and save 2D projections
        np.save(self.outputdir+"dmap_"+str(subN)+".npy", data)
        
        if (ps):
            pslists=auto_powerspectrum(data, Lbox=Lsub)
            np.save(self.outputdir+"pslists_"+str(subN)+".npy", pslists)
        
        hmap=cube_to_healpix(data, self.nside)
        np.save(self.outputdir+"hmap_"+str(subN)+".npy", hmap)
        print subN
                
