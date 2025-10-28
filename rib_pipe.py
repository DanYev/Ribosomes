import os
import sys
import shutil
import numpy as np
import MDAnalysis as mda
from pathlib import Path
from reforge import cli, io, mdm
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import *


def setup(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files and prepare directories
    mdsys.prepare_files() # be careful it can overwrite later files
    mdsys.sort_input_pdb(f"ribosome.pdb") # sorts chain and atoms in the input file and returns makes mdsys.inpdb file

    # # 1.2.1 Try to clean the input PDB and split the chains based on the type of molecules (protein, RNA/DNA)
    # mdsys.clean_pdb_mm(add_missing_atoms=False, add_hydrogens=True, pH=7.0)
    # mdsys.split_chains()
    # mdsys.clean_chains_mm(add_missing_atoms=False, add_hydrogens=False, pH=7.0)  # if didn't work for the whole PDB
    
    # 1.2.2 Same but if we want Go-Model for the proteins
    mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput='8\n 7\n', ignh='no', renum='yes') # 8 for CHARMM, sometimes you need to refer to AMBER FF
    mdsys.split_chains()
    mdsys.clean_chains_gmx(clinput='8\n 7\n', ignh='yes', renum='yes')
    mdsys.get_go_maps(append=True)

    # 1.3. COARSE-GRAINING. Done separately for each chain. If don't want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=1000, el=0.3, eu=0.9, p='backbone', pf=500, append=True)  # Martini + Elastic network FF 
    mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p='backbone', pf=500, append=True) # Martini + Go-network FF
    mdsys.martinize_rna(elastic='no', ef=50, el=0.5, eu=1.3, p='backbone', pf=500, append=True) # Martini RNA FF 
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    mdsys.make_box(d='1.25', bt='dodecahedron')
    
    # # 1.4. Coarse graining is *hopefully* done. Need to add solvent and ions
    solvent = os.path.join(mdsys.root, 'water.gro')
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius='0.21') # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.15, pname='K', nname='CL')

    # 1.5. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])
    
    
def md(sysdir, sysname, runname, ntomp): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    # mdrun.empp()
    # mdrun.mdrun(deffnm='em', ntomp=ntomp)
    # mdrun.eqpp(c='em.gro', r='em.gro', maxwarn='1') 
    # mdrun.mdrun(deffnm='eq', ntomp=ntomp)
    mdrun.mdpp(maxwarn='1')
    mdrun.mdrun(deffnm='md', ntomp=ntomp, bonded='gpu')
    
    
def extend(sysdir, sysname, runname, ntomp):    
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.mdrun(deffnm='md', cpi='md.cpt', ntomp=ntomp, nsteps=-2, bonded='gpu') # 'ext' for the bugged runs


def make_ndx(sysdir, sysname, **kwargs):
    system = GmxSystem(sysdir, sysname)
    system.make_system_ndx(backbone_atoms=["BB", "BB2"])
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault('b', 0) # in ps
    kwargs.setdefault('dt', 200) # in ps
    kwargs.setdefault('e', 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    # # For the bugged runs
    # with cd(mdrun.rundir):
    #     cli.gmx('trjcat', clinput='c\nc\n', cltext=True, f='md_old.trr ext.trr', o='md.trr', settime='yes')
    # Cov analysis
    k = 1 # k=1 to remove solvent, k=2 for backbone analysis.
    mdrun.convert_tpr(clinput=f'{k}\n', s='md.tpr', n=mdrun.sysndx, o='conv.tpr')
    mdrun.trjconv(clinput=f'{k}\n {k}\n', s='md.tpr', f='md.trr', o='conv.xtc', n=mdrun.sysndx, pbc='cluster', ur='compact', **kwargs)
    mdrun.trjconv(clinput='0\n 0\n', s='conv.tpr', f='conv.xtc', o='mdc.pdb', fit='rot+trans', e=0)
    mdrun.trjconv(clinput='0\n 0\n', s='conv.tpr', f='conv.xtc', o='mdc.xtc', fit='rot+trans')
    # Ions
    # k = 4 # k=4 to include ions
    # mdrun.convert_tpr(clinput=f'{k}\n', s='md.tpr', n=mdrun.sysndx, o='convi.tpr')
    # mdrun.trjconv(clinput=f'{k}\n {k}\n ', s='md.tpr', f='md.trr', n=mdrun.sysndx, o='convi.xtc', # trr from main runs
    #     pbc='cluster', **kwargs)
    # mdrun.trjconv(clinput='0\n 0\n', s='convi.tpr', f='convi.xtc', o='mdci.pdb', fit='rot+trans', e=0)
    # mdrun.trjconv(clinput='0\n 0\n', s='convi.tpr', f='convi.xtc', o='mdci.xtc', fit='rot+trans')
    # clean_dir(mdrun.rundir)


def rdf_analysis(sysdir, sysname, runname, **kwargs):
    system = GmxSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    rdfndx = os.path.join(system.wdir, 'rdf.ndx')
    ions = ['MG', 'K', 'CL']
    b = 400000
    for ion in ions:
        mdrun.rdf(clinput=f'BB1\n {ion}\n', n=rdfndx, o=f'rms_analysis/rdf_{ion}.xvg', 
            b=b, rmax=10, bin=0.01, **kwargs)

    
def rms_analysis(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault('b',  400000) # in ps
    kwargs.setdefault('dt', 200) # in ps
    kwargs.setdefault('e', 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    clean_dir(mdrun.rmsdir, '*npy')
    # Normal
    # mdrun.rmsf(clinput=f'2\n 2\n', s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit='yes', res='yes', **kwargs) 
    # mdrun.get_rmsf_by_chain(**kwargs)
    # mdrun.rmsd(clinput=f'2\n 2\n', s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit='rot+trans', **kwargs)
    # mdrun.get_rmsd_by_chain(b=0, **kwargs)
    # u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    # ag = u.atoms.select_atoms("name BB or name BB2")
    # positions = io.read_positions(u, ag, b=500000, e=3500000, sample_rate=1)
    # mdm.calc_and_save_rmsf(positions, outdir=mdrun.rmsdir, n=20)
    # For ions
    mdrun.rmsf(clinput=f'0\n 0\n', s=mdrun.rundir / 'convi.tpr', f=mdrun.rundir / 'mdci.xtc', fit='yes', res='no', **kwargs) 
    

def cluster(sysdir, sysname, runname, **kwargs):
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.cluster(clinput=f'0\n 0\n', b=500000, e=10000000, dt=20000, 
        cutoff=0.2, method='linkage', av='yes', 
        cl='clusters.xtc', clndx='cluster.ndx', )
    mdrun.extract_cluster()


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
    # mdrun.covar(clinput=f'0\n 0\n', f=f, s=s, ref='yes', last=10, b=b1, e=b1+dt, ascii='covar_1.dat', o='eigenval_1.xvg', v='eigenvec_1.trr', av='av_1.pdb', l='covar_1.log')
    # mdrun.covar(clinput=f'0\n 0\n', f=f, s=s, ref='yes', last=10, b=b2, e=b2+dt, ascii='covar_2.dat', o='eigenval_2.xvg', v='eigenvec_2.trr', av='av_2.pdb', l='covar_2.log')
    # mdrun.anaeig(v='eigenvec_1.trr',  v2='eigenvec_2.trr', over='overlap.xvg', **kwargs) #  inpr='inprod.xpm',
    # mdrun.covar(clinput=f'0\n 0\n', last=100)
    # mdrun.make_edi(clinput=f'1\n', s='../md.tpr', n=system.bbndx, radacc='1-3', slope=0.01, outfrq=10000, o='../sam.edi')
    
    
def overlap(sysdir, sysname, **kwargs):
    system = GmxSystem(sysdir, sysname)
    run1 = system.initmd('mdrun_2')
    run2 = system.initmd('mdrun_4')
    run3 = system.initmd('mdrun_5')
    v1 = os.path.join(run1.covdir, 'eigenvec.trr')
    v2 = os.path.join(run2.covdir, 'eigenvec.trr')
    v3 = os.path.join(run3.covdir, 'eigenvec.trr')
    run1.anaeig(v=v1, v2=v2, over='overlap_1.xvg', **kwargs)
    run1.anaeig(v=v2, v2=v3, over='overlap_2.xvg', **kwargs)
    run1.anaeig(v=v3, v2=v1, over='overlap_3.xvg', **kwargs)


def cov_analysis(sysdir, sysname, runname):
    mdrun = GmxRun(sysdir, sysname, runname)
    s, f = mdrun.rundir / 'mdc.pdb', mdrun.rundir / 'mdc.xtc'
    u = mda.Universe(s, f, in_memory=True)
    # u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    # Select the backbone atoms. AA: "name CA or name P or name C1'" CG: "name BB or name BB1 or name BB3"
    backbone_anames = "name BB or name BB2"
    ag = u.atoms.select_atoms(backbone_anames)
    clean_dir(mdrun.covdir, '*npy')
    mdrun.get_covmats(u, ag, sample_rate=1, b=400000, e=10000000, n=70, outtag='covmat') 
    mdrun.get_pertmats()
    mdrun.get_dfi(outtag='dfi')
    mdrun.get_dci(outtag='dci', asym=False)
    mdrun.get_dci(outtag='asym', asym=True)
    # Calc DCI between chains
    atoms = io.pdb2atomlist(mdrun.solupdb)
    backbone_anames = ["BB", "BB2"]
    bb = atoms.mask(backbone_anames, mode='name')
    bb.renum() # Renumber atids form 0, needed to mask numpy arrays
    groups = bb.chains.atids # mask for the arrays
    labels = [chids[0] for chids in bb.chains.chids]
    mdrun.get_group_dci(groups=groups, labels=labels, asym=False, outtag='dci')
    mdrun.get_group_dci(groups=groups, labels=labels, asym=False, transpose=True, outtag='tdci')
    mdrun.get_group_dci(groups=groups, labels=labels, asym=True, outtag='asym')


def pockets(sysdir, sysname, runname):
    mdrun = GmxRun(sysdir, sysname, runname)
    s, f = mdrun.rundir / 'mdc.pdb', mdrun.rundir / 'mdc.xtc'
    # u = mda.Universe(s, f, in_memory=True)
    # atoms = io.pdb2atomlist(mdrun.solupdb)
    spc = [4215, 4216, 4217, 4218, 4219, 4220, 4221, 4345, 4346, 4347]
    ptc_wt = [703, 704, 705, 1800, 1801, 2160, 2161, 2162, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2380, 2381, 2568, 2570, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2681, 2682, 2683, 2684, 2685, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2730, 2731, 2732, 2733, 2737, 2738, 2739, 2740, 2741, 4784, 4785, 4786, 4937, 4938, 4939, 4940, 5375, 8560, 8561, 8562, 8563, 8565, 8569, 8570, 8571, 8572, 8573, 8574, 8575, 8576, 8577, 8578, 8579, 8580]
    ptc_dl11 = [5375, 5376, 4937, 4784, 4938, 4785, 4939, 4786, 4940, 8417, 8418, 8421, 8422, 8423, 8424, 8425, 8426, 8427, 8429, 8430, 8431, 8432, 8433, 8434, 8435, 8436, 8437, 8438, 8439, 8440, 8441, 8442, 703, 704, 1259, 1800, 1801, 2160, 2161, 2162, 2163, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2380, 2568, 2570, 2575, 2576, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2619, 2620, 2621, 2622, 2623, 2624, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2681, 2682, 2683, 2684, 2685, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2730, 2731, 2732, 2733, 2737, 2738, 2739, 2740, 2741]
    rela_wt = [1011, 1012, 1013, 1014, 1015, 1016, 1022, 1023, 1024, 1025, 1026, 1193, 1194, 1195, 1202, 1203, 1204, 1205, 1207, 1208, 4107, 4108, 4109, 4110, 4111, 4137, 4138, 4882, 4917, 4919, 4920, 4921, 4922, 7840, 7842, 7843, 7844, 7861, 7863, 7866, 8105, 8109, 9653, 9654, 9655, 9656, 9657, 9703, 9704, 9705, 9706]
    rela_dl11 = [4882, 7834, 7835, 7836, 7840, 4917, 7842, 4919, 7843, 4920, 7844, 9514, 4921, 9515, 7845, 4922, 9516, 9517, 9518, 7860, 7861, 7862, 7863, 7865, 7866, 9564, 9565, 9566, 9567, 1011, 1012, 1015, 1016, 1022, 1023, 1024, 1026, 4107, 4108, 4109, 4110, 4111, 4137, 4138, 4139, 1193, 1194, 1202, 1205, 1206, 1207, 1209]
    in_pdb = 'pdb/obg_dL11.pdb'
    atoms = io.pdb2atomlist(in_pdb)
    pocket = [x-1 for x in atoms.atids]
    groups = [pocket]
    labels = ['obg']
    mdrun.get_group_dci(groups=groups, labels=labels, asym=False, outtag='dci')
    mdrun.get_group_dci(groups=groups, labels=labels, asym=False, transpose=True, outtag='tdci')
    mdrun.get_group_dci(groups=groups, labels=labels, asym=True, outtag='asym')


def tdlrt_analysis(sysdir, sysname, runname):
    mdrun = GmxRun(sysdir, sysname, runname)
    # CCF params FRAMEDT=20 ps
    b = 0
    e = 100000
    sample_rate = 1
    ntmax = 1000 # how many frames to save
    fname = 'corr_pv.npy'
    corr_file = os.path.join(mdrun.lrtdir, fname)
    # CALC CCF
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms
    positions = io.read_positions(u, ag, sample_rate=sample_rate, b=b, e=e) 
    velocities = io.read_velocities(u, ag, sample_rate=sample_rate, b=b, e=e)
    corr = lrt.ccf(positions, velocities, ntmax=ntmax, n=5, mode='gpu', center=True)
    np.save(corr_file, corr)


def get_averages(sysdir, sysname):
    system = GmxSystem(sysdir, sysname)   
    # system.get_mean_sem(pattern='rmsf*.npy')
    # system.get_mean_sem(pattern='dfi*.npy')
    # system.get_mean_sem(pattern='ggdci*.npy') # group-group DCI    
    # system.get_mean_sem(pattern='dci*.npy')
    # system.get_mean_sem(pattern='asym*.npy')
    # for chain in system.chains:
    #     logger.info(f'Processing chain {chain}')
    #     system.get_mean_sem(pattern=f'gdci_{chain}*.npy')
    #     system.get_mean_sem(pattern=f'gtdci_{chain}*.npy')
    #     system.get_mean_sem(pattern=f'gasym_{chain}*.npy')
        # system.get_mean_sem(pattern=f'crmsf_{chain}*.npy')
    # Pockets
    pockets = ['obg']
    for pocket in pockets:
        logger.info(f'Processing pocket {pocket}')
        system.get_mean_sem(pattern=f'gdci_{pocket}*.npy')
        system.get_mean_sem(pattern=f'gtdci_{pocket}*.npy')
        system.get_mean_sem(pattern=f'gasym_{pocket}*.npy')


def get_td_averages(sysdir, sysname):
    system = GmxSystem(sysdir, sysname)  
    system.get_td_averages('pertmat*.npy')
            

def ajob(sysdir, sysname, runname):
    system = GmxSystem(sysdir, sysname)
    run = system.initmd(runname)
    # cli.run_gmx(run.rundir, 'convert-trj', f='md.trr', o='md_old.trr', e=1600, tu='ns')
    # cli.gmx_grompp(run.rundir, c=system.sysgro, p=system.systop, f=os.path.join(system.mdpdir, 'md.mdp'), t='md_old.trr',  o='ext.tpr')
    # cli.gmx_mdrun(run.rundir, s='ext.tpr', deffnm='ext')
    # cli.run_gmx(run.rundir, 'trjcat', clinput='c\nc\n', cltext=True, f='md_old.trr ext.trr', o='md.trr', settime='yes')


if __name__ == '__main__':
    command = sys.argv[1]
    args = sys.argv[2:]
    commands = {
        "setup": setup,
        "md": md,
        "extend": extend,
        "make_ndx": make_ndx,
        "trjconv": trjconv,
        "rms_analysis": rms_analysis,
        "cluster": cluster,
        "cov_analysis": cov_analysis,
        "tdlrt_analysis": tdlrt_analysis,
        "get_averages": get_averages,
        "get_td_averages": get_td_averages,
        "ajob": ajob,
        "viz": viz,
        "pockets": pockets,
    }
    if command in commands:   # Then, assuming `command` is the command name (a string)
        commands[command](*args) # `args` is a list/tuple of arguments
    else:
        raise ValueError(f"Unknown command: {command}")
        
    