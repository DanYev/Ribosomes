from pathlib import Path
import MDAnalysis as mda
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun, get_ntomp
from reforge.utils import clean_dir, get_logger

logger = get_logger()

# Global settings
INPDB = 'input.pdb'
dt = 0.020  # Time step in picoseconds
total_time = 1000  # Total simulation time in nanoseconds
NSTEPS = int(total_time * 1e3 / dt)  # Number of MD steps for production run

def workflow(sysdir, sysname, runname):
    setup_martini(sysdir, sysname)
    md_npt(sysdir, sysname, runname)
    trjconv(sysdir, sysname, runname)


def setup(*args):
    setup_martini(*args)


def setup_martini(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)
    inpdb = mdsys.root / INPDB
    # 1.1. Need to copy force field and md-parameter files and prepare PDBs and directories
    mdsys.prepare_files(pour_martini=True) # be careful it can overwrite later files
    mdsys.clean_pdb_mm(inpdb, add_missing_atoms=False, add_hydrogens=False, pH=7.0) # Generates Amber ff names in PDB
    # mdsys.clean_pdb_gmx(inpdb, clinput="8\n 7\n", ignh="no", renum="yes") # 8 for CHARMM, sometimes you need to refer to AMBER FF
    mdsys.split_chains()
    
    # # 1.2.2 Looks like we don't need this anymore
    # mdsys.get_go_maps(append=True)

    # 1.2. COARSE-GRAINING. Done separately for each chain. If don"t want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=1000, el=0.3, eu=0.9, from_ff='charmm', p="backbone", pf=500, append=False)  # Martini + Elastic network FF 
    mdsys.martinize_proteins_go(go_eps=12.0, go_low=0.3, go_up=1.0, from_ff='amber', p="backbone", pf=500, append=False) # Martini + Go-network FF
    mdsys.martinize_rna(elastic="yes", ef=100, el=0.5, eu=1.2, merge=True, p="backbone", pf=500, append=False) # Martini RNA FF 
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    
    # 1.3. Coarse graining is *hopefully* done. Need to add solvent and ions
    mdsys.make_box(d="1.2", bt="dodecahedron")
    solvent = mdsys.root / "water.gro"
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17") # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")

    # 1.4. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])
    
    
def md_npt(sysdir, sysname, runname): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    ntomp = get_ntomp()
    mdrun.empp(f=mdrun.mdpdir / "em_cg.mdp")
    mdrun.mdrun(deffnm="em", ntomp=ntomp)
    mdrun.hupp(f=mdrun.mdpdir / "hu_cg.mdp", c="em.gro", r="em.gro", maxwarn="1") 
    mdrun.mdrun(deffnm="hu", ntomp=ntomp)
    mdrun.eqpp(f=mdrun.mdpdir / "eq_cg.mdp", c="hu.gro", r="hu.gro", maxwarn="1") 
    mdrun.mdrun(deffnm="eq", ntomp=ntomp)
    mdrun.mdpp(f=mdrun.mdpdir / "md_cg.mdp", maxwarn="1")
    mdrun.mdrun(deffnm="md", ntomp=ntomp, nsteps=NSTEPS, ) # bonded="gpu")
    
    
def extend(sysdir, sysname, runname):    
    mdrun = GmxRun(sysdir, sysname, runname)
    ntomp = get_ntomp()
    dt = 0.020 # picoseconds
    t_ext = 10000 # nanoseconds
    nsteps = int(t_ext * 1e3 / dt)
    mdrun.mdrun(deffnm="md", cpi="md.cpt", ntomp=ntomp, nsteps=NSTEPS, ) # bonded="gpu") 
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault("b", 0) # in ps
    kwargs.setdefault("dt", 200) # in ps
    kwargs.setdefault("e", 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    k = 1 # k=1 to remove solvent, k=2 for backbone analysis, k=4 to include ions
    # mdrun.trjconv(clinput=f"0\n 0\n", s="eq.tpr", f="eq.gro", o="viz.pdb", n=mdrun.sysndx, pbc="atom", ur="compact", e=0)
    mdrun.convert_tpr(clinput=f"{k}\n", s="md.tpr", n=mdrun.sysndx, o="topology.tpr")
    mdrun.trjconv(clinput=f"{k}\n {k}\n", s="md.tpr", f="md.xtc", o="conv.xtc", n=mdrun.sysndx, pbc="cluster", ur="compact", **kwargs)
    mdrun.trjconv(clinput="0\n 0\n", s="topology.tpr", f="conv.xtc", o="topology.pdb", fit="rot+trans", e=0)
    mdrun.trjconv(clinput="0\n 0\n", s="topology.tpr", f="conv.xtc", o="samples.xtc", fit="rot+trans")
    clean_dir(mdrun.rundir)


def nm_analysis(sysdir, sysname, runname, **kwargs):
    system = GmxSystem(sysdir, sysname)
    mdrun = GmxRun(sysdir, sysname, runname)
    # f = os.path.join(mdrun.cludir, 'trajout_Cluster_0001.xtc')
    # s = os.path.join(mdrun.cludir, 'clusters.pdb')
    f = '../traj.xtc'
    s = mdrun.trjpdb
    b = 000000
    mdrun.covar(clinput=f'0\n 0\n', b=b, f=f, s=s, ref='no', last=1000, ascii='covar.dat', o='eigenval.xvg', v='eigenvec.trr', av='av.pdb', l='covar.log')
    mdrun.anaeig(clinput=f'0\n 0\n', b=b, dt=15000, first=1, last=10, filt='filtered.pdb', v='eigenvec.trr', eig='eigenval.xvg', proj='proj.xvg') 
    b1 = 000000
    b2 = 150000
    dt = 150000
    # mdrun.covar(clinput=f'0\n 0\n', f=f, s=s, ref='yes', last=10, b=b1, e=b1+dt, ascii='covar_1.dat', o='eigenval_1.xvg', v='eigenvec_1.trr', av='av_1.pdb', l='covar_1.log')
    # mdrun.covar(clinput=f'0\n 0\n', f=f, s=s, ref='yes', last=10, b=b2, e=b2+dt, ascii='covar_2.dat', o='eigenval_2.xvg', v='eigenvec_2.trr', av='av_2.pdb', l='covar_2.log')
    # mdrun.anaeig(v='eigenvec_1.trr',  v2='eigenvec_2.trr', over='overlap.xvg', **kwargs) #  inpr='inprod.xpm',
    # mdrun.covar(clinput=f'0\n 0\n', last=100)
    # mdrun.make_edi(clinput=f'1\n', s='../md.tpr', n=system.bbndx, radacc='1-3', slope=0.01, outfrq=10000, o='../sam.edi')

if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()

    