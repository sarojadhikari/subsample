# use mpi4py for subsampling

from mpi4py import MPI
from subsample import subsample, subsubsample
import sys

argc=len(sys.argv)
name=sys.argv[1]
seed=sys.argv[2]

if (argc>3):
    NFILES=int(sys.argv[3])
    LMESH=int(sys.argv[4])
    SUBX=int(sys.argv[5])
    Lbox=int(sys.argv[6])
else:
    NFILES=32
    LMESH=2048
    SUBX=4
    Lbox=80000

ss = subsample(filebase="pot.delta", Nfiles=NFILES, Lmesh=LMESH, subx=SUBX)
bdir="/gpfs/home/sza5154/scratch/"+name+"/"+seed+"/"
ss.set_basedir(bdir)
ss.set_outputdir(ss.basedir+"64/")


newsubx=2
sss=subsubsample(Lmesh=LMESH/SUBX, subx=newsubx)
sss.set_basedir(bdir+"64/")
sss.set_outputdir(bdir+"512/")

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
    ss.GenerateSubSample(si, sx, sy, ps=True, Lsub=Lbox/ss.subx)
    #print Lbox/ss.subx
    comm.Barrier()

# code to further breakdown each subsample into newsubx^3 subvolumes
# this way we get both 4^3=64 and 64*8=512 subsamples at once

for i in range(0, bdown):
    sampnum=comm.rank+(i*NFILES)
    sss.samplenumber=sampnum
    sss.GenerateSSSamples(ps=True, Lsub=Lbox/SUBX/newsubx)
    comm.Barrier()

if (comm.rank==0):
    print "done"