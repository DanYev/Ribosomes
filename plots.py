import os
import numpy as np
import pandas as pd
import sys
from reforge import io, mdm
from reforge.mdsystem import gmxmd
from reforge.plotting import *
from reforge.utils import logger


def pull_data(metric):
    files = io.pull_files(system.datdir, metric)
    datas = [np.load(file) for file in files if '_av' in file]
    errs = [np.load(file) for file in files if '_err' in file]
    return datas, errs


def set_bfactors_by_residue(in_pdb, bfactors, out_pdb=None):
    atoms = io.pdb2atomlist(in_pdb)
    residues = atoms.residues
    for idx, residue in enumerate(residues):
        for atom in residue:
            atom.bfactor = bfactors[idx]
    if out_pdb:
        atoms.write_pdb(out_pdb)
    return atoms


def set_bfactors_by_atom(in_pdb, bfactors, out_pdb=None):
    atoms = io.pdb2atomlist(in_pdb)
    for idx, atom in enumerate(atoms):
        atom.bfactor = bfactors[idx]
    if out_pdb:
        atoms.write_pdb(out_pdb)
    return atoms


def set_ax_parameters(ax, xlabel=None, ylabel=None, axtitle=None):
    """
    ax - matplotlib ax object
    """
    # Set axis labels and title with larger font sizes
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(axtitle, fontsize=16)
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5, width=1.5)
    # Increase spine width for a bolder look
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Add a legend with custom font size and no frame
    legend = ax.legend(fontsize=14, frameon=False)
    # Optionally, add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)


def get_segments():
    # Calc DCI between chains
    atoms = io.pdb2atomlist(system.solupdb)
    backbone_anames = ["BB", "BB2"]
    bb = atoms.mask(backbone_anames, mode='name')
    bb.renum() # Renumber atids form 0, needed to mask numpy arrays
    groups = bb.chains.atids # mask for the arrays
    labels = [chids[0] for chids in bb.chains.chids]
    for label, group in zip(labels, groups):
        print(label, group[-1])
    return labels



def plot_dfi(system):
    # Pulling data
    datas, errs = pull_data('dfi*')
    param = {'lw':2}
    xs = [np.arange(len(data)) for data in datas]
    datas = [data*len(data) for data in datas]
    errs = [err*len(err) for err in errs]
    params = [param for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DFI')
    plot_figure(fig, ax, figname=system.sysname, figpath='png/dfi.png',)


def plot_pdfi(system):
    # Pulling data
    datas, errs = pull_data('dfi*')
    param = {'lw':2}
    xs = [np.arange(len(data)) for data in datas]
    datas = [percentile(data) for data in datas]
    params = [param for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='%DFI')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath='png/pdfi.png',)


def plot_rmsf(system):
    # Pulling data
    datas, errs = pull_data('crmsf_B*')
    xs = [np.arange(len(data)) for data in datas]
    datas = [data*10 for data in datas]
    errs = [err*10 for err in errs]
    params = [{'lw':2} for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='RMSF (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath='png/rmsf.png',)


def plot_rmsd(system):
    # Pulling data
    files = io.pull_files(system.mddir, 'rmsd*npy')
    datas = [np.load(file) for file in files]
    labels = [file.split('/')[-3] for file in files]
    xs = [data[0]*1e-3 for data in datas]
    ys = [data[1]*10 for data in datas]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, ys, params)
    set_ax_parameters(ax, xlabel='Time (ns)', ylabel='RMSD (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper() , figpath=system.pngdir / 'rmsd.png',)


def plot_dci(system):
    # Pulling data
    datas, errs = pull_data('ggdci*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # data = mdm.dci(data)
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=0, vmax=3)
    labels = get_segments()
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)   # now the x-axis shows your strings
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname='DCI', figpath='png/dci.png',)


def plot_asym(system):
    # Pulling data
    datas, errs = pull_data('asym*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=-1, vmax=1)
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname='DCI asymmetry', figpath='png/asym.png',)


def make_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_av.npy'))
    err = np.load(os.path.join(system.datdir, f'{label}_err.npy'))
    if factor:
        data *= factor
        err *= factor
    data_pdb = os.path.join(system.pngdir, f'{label}.pdb')
    err_pdb = os.path.join(system.pngdir, f'{label}_err.pdb')
    set_bfactors_by_residue(system.inpdb, data, data_pdb)
    set_bfactors_by_residue(system.inpdb, err, err_pdb)


def make_cg_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_av.npy'))
    err = np.load(os.path.join(system.datdir, f'{label}_err.npy'))
    if factor:
        data *= factor
        err *= factor
    data_pdb = os.path.join(system.pngdir, f'{label}.pdb')
    err_pdb = os.path.join(system.pngdir, f'{label}_err.pdb')
    set_bfactors_by_atom(system.root / 'mdci.pdb', data, data_pdb)
    set_bfactors_by_atom(system.root / 'mdci.pdb', err, err_pdb)


def make_enm_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_enm.npy'))
    if factor:
        data *= factor
    data_pdb = os.path.join(system.pngdir, f'enm_{label}.pdb')
    set_bfactors_by_residue(system.inpdb, data, data_pdb)


def make_delta_pdb(system_1, system_2, label, out_name, filter=True, factor=None):
    logger.info('Making Delta PDB')
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy'))  
    if factor:
        data_1 *= factor
        err_1 *= factor
        data_2 *= factor
        err_2 *= factor
    data = data_1 - data_2
    err = np.sqrt(err_1**2 + err_2**2)
    if filter:
        mask = np.abs(data) < 2.0 * err
        data[mask] = 0
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    err_pdb = os.path.join('systems', 'pdb', out_name + '_err.pdb')
    set_bfactors_by_residue(system_1.inpdb, data, data_pdb)
    set_bfactors_by_residue(system_1.inpdb, err, err_pdb) 
    logger.info('Saved Delta PDB to %s', data_pdb)


def make_delta_cg_pdb(system_1, system_2, label, out_name, factor=None):
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy'))  
    if factor:
        data_1 *= factor
        err_1 *= factor
        data_2 *= factor
        err_2 *= factor
    data = data_1 - data_2
    err = np.sqrt(err_1**2 + err_2**2)
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    err_pdb = os.path.join('systems', 'pdb', out_name + '_err.pdb')
    set_bfactors_by_atom(system.root / 'mdci.pdb', data, data_pdb)
    set_bfactors_by_atom(system.root / 'mdci.pdb', err, err_pdb)

def rmsf_pdb(system):
    logger.info(f'Making RMSF PDB')
    make_cg_pdb(system, 'rmsf', factor=10)


def dfi_pdb(system):
    logger.info(f'Making DFI PDB')
    make_pdb(system, 'dfi')


def dci_pdbs(system):
    chains = ['A', 'k']
    # chains = system.chains
    for chain in chains:
        logger.info(f'Making DCI {chain} PDB')
        label = f'gdci_{chain}'
        make_pdb(system, label)


def pocket_dci_pdbs(system):
    pockets = ['obg']
    for chain in pockets:
        logger.info(f'Making DCI {chain} PDB')
        label = f'gdci_{chain}'
        make_pdb(system, label)


def runs_metric(system, metric):
    files = io.pull_files(system.mddir, metric)
    files = [f for f in files if '.npy' in f]
    datas = [np.load(file) for file in files]
    xs = [np.arange(len(data)) for data in datas]
    datas = [data for data in datas]
    params = [{'lw':2, 'label':fname} for data, fname in zip(datas, files)]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='RMSF (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath='png/metric.png',)


def plot_delta_heatmap(system_1, system_2, label, out_name='ddci.png', filter=True, factor=None):
    logger.info('Making Delta Heatmap')
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy'))  
    # data_1 = mdm.dci(data_1)
    # data_2 = mdm.dci(data_2)
    if factor:
        data_1 *= factor
        err_1 *= factor
        data_2 *= factor
        err_2 *= factor
    data = data_1 - data_2
    err = np.sqrt(err_1**2 + err_2**2)
    if filter:
        mask = np.abs(data) < 1.0 * err
        data[mask] = 0 
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=-0.5, vmax=0.5)
    labels = get_segments()
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)   # now the x-axis shows your strings
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    figpath = f'png/{out_name}'
    plot_figure(fig, ax, figname='DCI', figpath=figpath,)    
    logger.info('Saved figure to %s', figpath)


if __name__ == '__main__':
    sysdir = 'systems' 
    sysnames = ['ribosome_mgh', 'ribosome_mg', 'ribosome_wt']
    for sysname in sysnames:
        system = gmxmd.GmxSystem(sysdir, sysname)
        dci_pdbs(system)
    # get_segments()
    # plot_dfi(system)
    # plot_pdfi(system)
    # plot_ggdci(system)
    # plot_dci(system)
    # plot_asym(system)
    # plot_rmsf(system)
    # plot_rmsd(system)
    # # # PDBs
    # rmsf_pdb(system)
    # dfi_pdb(system)
    # pocket_dci_pdbs(system)
    # make_enm_pdb(system, label='dfi')
    # Deltas
    variant_1 = 'mg'
    variant_2 = 'wt'
    label = 'ggdci'
    system_1 = gmxmd.GmxSystem(sysdir, f'ribosome_{variant_1}')
    system_2 = gmxmd.GmxSystem(sysdir, f'ribosome_{variant_2}')
    # make_delta_pdb(system_1, system_2, label=label, out_name=f'{label}_{variant_1}_{variant_2}', filter=True)
    # make_delta_cg_pdb(system_1, system_2, label='rmsf', out_name=f'drmsf_{variant_1}_{variant_2}')
    plot_delta_heatmap(system_1, system_2, label=label, out_name='ddci.png', filter=True, factor=None)
    # runs_metric(system, 'rmsf*')





