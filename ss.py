# use mpi4py for subsampling

from mpi4py import MPI
from subsample import subsample
import sys

argc=len(sys.argv)
name=sys.argv[1]
seed=sys.argv[2]

if (argc>3):
    NFILES=int(sys.argv[3])
    LMESH=int(sys.argv[4])
    SUBX=int(sys.argv[5])
    nside=int(sys.argv[6])
else:
    NFILES=32
    LMESH=2048
    SUBX=4
    nside=256

ss = subsample(filebase="gauspot.delta", Nfiles=NFILES, Lmesh=LMESH, subx=SUBX, NSIDE=nside)
ss.set_basedir("/gpfs/home/sza5154/scratch/"+name+"/"+seed+"/")
ss.set_outputdir(ss.basedir)

comm = MPI.COMM_WORLD
print "number of cores: "+str(comm.size)+"\n"
# initialize the subsampling code
# first generate segments
ss.GenerateSegments(comm.rank)
#print comm.rank
comm.Barrier() # end till everything is done

# now distribute the generation of subsamples (64 of them) in the 32 cores
segs=ss.subx
bdown=int(ss.Nsubs/ss.Nfiles)

# generate si, sx, sy from cNum, cpus, and segs

def sxy(cNum, segs):
    si = int(cNum/(segs*segs))
    sx = int((cNum%(segs*segs))/segs)
    sy = int((cNum%(segs*segs))%segs)
    return [si, sx, sy]

for i in range(0, bdown):
    si, sx, sy = sxy((i*ss.Nfiles)+comm.rank, segs)
    ss.GenerateSubSample(si, sx, sy)
    comm.Barrier()

if (comm.rank==0):
    print "done"
            

