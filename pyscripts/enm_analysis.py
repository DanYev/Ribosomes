from pathlib import Path
import numpy as np
from reforge import mdm, io
from reforge.mdsystem.mdsystem import MDSystem
from reforge.utils import logger


sysdir = 'systems'
sysname = 'ribosome_wt' 
system = MDSystem(sysdir, sysname)
in_pdb = system.inpdb
atoms = io.pdb2atomlist(in_pdb)
backbone_anames = ["CA", "C1'", "MG", "ZN"]
bb = atoms.mask(backbone_anames, mode='name')
vecs = np.array(bb.vecs)
logger.info("Caclulating Hessian")
hess = mdm.hessian(vecs, cutoff=100, dd=6)
logger.info("Inverting Hessian")
invhess = mdm.inverse_matrix(hess, device="gpu_sparse", k_singular=6, n_modes=100, dtype=np.float32)
logger.info("Calculating Perturbation Matrix")
pertmat = mdm.perturbation_matrix_iso(invhess.get().astype(np.float32))
# invhess /= np.sqrt(np.average(invhess**2))
pertmat_npy = system.datdir / "pertmat_enm.npy"
np.save(pertmat_npy, pertmat)
dfi = mdm.dfi(pertmat)
dfi_npy = system.datdir / "dfi_enm.npy"
np.save(dfi_npy, dfi)
    


