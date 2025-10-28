import os
import numpy as np
import pandas as pd
import sys
from reforge import io
from reforge.mdsystem import gmxmd
from reforge.plotting import *
from reforge.utils import logger
from reforge.mdm import percentile

ptc_wt = [703, 704, 705, 1800, 1801, 2160, 2161, 2162, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2380, 2381, 2568, 2570, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2681, 2682, 2683, 2684, 2685, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2730, 2731, 2732, 2733, 2737, 2738, 2739, 2740, 2741, 4784, 4785, 4786, 4937, 4938, 4939, 4940, 5375, 8560, 8561, 8562, 8563, 8565, 8569, 8570, 8571, 8572, 8573, 8574, 8575, 8576, 8577, 8578, 8579, 8580]
ptc_dl11 = [5375, 5376, 4937, 4784, 4938, 4785, 4939, 4786, 4940, 8417, 8418, 8421, 8422, 8423, 8424, 8425, 8426, 8427, 8429, 8430, 8431, 8432, 8433, 8434, 8435, 8436, 8437, 8438, 8439, 8440, 8441, 8442, 703, 704, 1259, 1800, 1801, 2160, 2161, 2162, 2163, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2380, 2568, 2570, 2575, 2576, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2619, 2620, 2621, 2622, 2623, 2624, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2681, 2682, 2683, 2684, 2685, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2730, 2731, 2732, 2733, 2737, 2738, 2739, 2740, 2741]
rela_wt = [1011, 1012, 1013, 1014, 1015, 1016, 1022, 1023, 1024, 1025, 1026, 1193, 1194, 1195, 1202, 1203, 1204, 1205, 1207, 1208, 4107, 4108, 4109, 4110, 4111, 4137, 4138, 4882, 4917, 4919, 4920, 4921, 4922, 7840, 7842, 7843, 7844, 7861, 7863, 7866, 8105, 8109, 9653, 9654, 9655, 9656, 9657, 9703, 9704, 9705, 9706]
rela_dl11 = [4882, 7834, 7835, 7836, 7840, 4917, 7842, 4919, 7843, 4920, 7844, 9514, 4921, 9515, 7845, 4922, 9516, 9517, 9518, 7860, 7861, 7862, 7863, 7865, 7866, 9564, 9565, 9566, 9567, 1011, 1012, 1015, 1016, 1022, 1023, 1024, 1026, 4107, 4108, 4109, 4110, 4111, 4137, 4138, 4139, 1193, 1194, 1202, 1205, 1206, 1207, 1209]


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


def get_atids_from_pdb(in_pdb):
    atoms = io.pdb2atomlist(in_pdb)
    atids = atoms.atids
    atids = list(map(lambda x: x - 1, atids))
    return atids


def get_metric_diff_for_pdb(in_pdb='pyscripts/l12_arm.pdb', metric='dfi'):
    sysdir = 'systems'
    system_wt = gmxmd.GmxSystem(sysdir, 'ribosome_wt')
    system_mut = gmxmd.GmxSystem(sysdir, 'ribosome_dL11')

    atoms = io.pdb2atomlist(in_pdb)
    pocket = [x-1 for x in atoms.atids]

    metric_wt = np.load(system_wt.root / 'data' / f'{metric}_av.npy')
    metric_mut = np.load(system_mut.root / 'data' / f'{metric}_av.npy')
    err_wt = np.load(system_wt.root / 'data' / f'{metric}_err.npy')
    err_mut = np.load(system_mut.root / 'data' / f'{metric}_err.npy')

    metric_wt = metric_wt[pocket] 
    metric_mut = metric_mut[pocket] 
    err_wt = err_wt[pocket] 
    err_mut = err_mut[pocket]
    data = metric_mut - metric_wt
    err = np.sqrt(err_mut**2 + err_wt**2)
    return data, err
    
 
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

   
def make_delta_pdb_for_dL11(system_1, system_2, label, out_name):
    logger.info('DOING DL11 PDB')
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy'))   
    if 0:
        data_1 *= len(data_1)
        err_1 *= len(err_1)
        data_2 *= len(data_2)
        err_2 *= len(err_2)  
    atoms = io.pdb2atomlist(system_2.inpdb)
    mask = [residue.chids[0] != 'k' for residue in atoms.residues]
    mask.append(True)
    data_2 = data_2[mask]
    err_2 = err_2[mask]
    data = data_1 - data_2
    err = np.sqrt(err_1**2 + err_2**2) 
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    err_pdb = os.path.join('systems', 'pdb', out_name + '_err.pdb')
    set_bfactors_by_residue(system_1.inpdb, data, data_pdb)
    set_bfactors_by_residue(system_1.inpdb, err, err_pdb)

   
def make_delta_enm_pdb_for_dL11(system_1, system_2, label, out_name):
    logger.info('DOING DL11 PDB')
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}.npy'))
    if 0:
        data_1 *= len(data_1)
        data_2 *= len(data_2)
    atoms = io.pdb2atomlist(system_2.inpdb)
    residues = atoms.residues
    # mask = [residue.chids[0] != 'I' for residue in residues] # for normal dl11
    mask = [residue.chids[0] not in ["I", "5", "7", "8"] for residue in residues] # for the dimer (I, 5, 7, 8)
    data_2 = data_2[mask]
    data = data_1 - data_2 
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    set_bfactors_by_residue(system_1.inpdb, data, data_pdb)

    
def make_delta_hm_for_dL11(system_1, system_2, label='ggdci'):
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    data_2 =  np.delete(np.delete(data_2, 36, axis=0), 36, axis=1)
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy')) 
    err_2 =  np.delete(np.delete(err_2, 36, axis=0), 36, axis=1)
    data = data_1 - data_2
    err = np.sqrt(err_1**2 + err_2**2) 
    # data
    fig, ax = init_figure(grid=(1, 1), axsize=(6, 6))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=-0.15, vmax=0.15)
    set_ax_parameters(ax, xlabel='Perturbed Residue', ylabel='Coupled Residue')
    plot_figure(fig, ax, figname='dDCI(WT-dL11)', figpath=system_1.sysdir / 'png' / 'ddci.png',)
    # data
    fig, ax = init_figure(grid=(1, 1), axsize=(6, 6))
    make_heatmap(ax, err, cmap='bwr', interpolation=None, vmin=-0.15, vmax=0.15)
    set_ax_parameters(ax, xlabel='Perturbed Residue', ylabel='Coupled Residue')
    plot_figure(fig, ax, figname='dDCI Err', figpath=system_1.sysdir / 'png' / 'ddci_err.png',)


def dfi_pdb(system_1, system_2, label, out_name):
    logger.info('DOING DL11 PDB')
    atoms = io.pdb2atomlist('systems/pdb/ribosome_dfi.pdb')
    atoms_k = atoms.mask_out("k", mode='chid')
    atoms_wt = atoms_k.mask(["C1'", "CA"], mode='name')
    data_1 = np.array(atoms_wt.bfactors)
    # mask = [residue.chids[0] != 'k' for residue in atoms_wt.residues]
    # data_1 = data_1[mask]
    atoms = io.pdb2atomlist('systems/pdb/delta_dfi_wt_dL11.pdb')
    atoms_l11 = atoms.mask(["C1'", "CA"], mode='name')
    data_2 = np.array(atoms_l11.bfactors)
    data = data_1 + data_2
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    set_bfactors_by_residue(system_2.inpdb, data, data_pdb)


def arms_ddfi_png(in_pdb, metric='dfi'):
    sysdir = 'systems'
    atoms = io.pdb2atomlist(in_pdb)
    pocket = [x-1 for x in atoms.atids]
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    y = np.array(atoms.bfactors)
    err_pdb = 'pyscripts/ddfi_wt_dL11_err.pdb'
    err_atoms = io.pdb2atomlist(err_pdb)
    err_pocket = [err_atoms[i] for i in pocket]
    err = np.array([atom.bfactor for atom in err_pocket])
    x = np.arange(len(y))
    ax.plot(x, y, label=r'$\Delta L11-WT$', lw=1.5, color='k')
    ax.fill_between(x, y - err, y + err, alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.3, 0.2)
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlabel(r'$Residue$', fontsize=12)
    ax.set_ylabel(r'L1 Arm $\Delta DFI$', fontsize=12)
    # ax.set_title('L7/12 Arm', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'png/ddfi_l1arm_dL11_wt.png')
    plt.close()


def pocket_ddfi_png(system_dL11, system_wt, metric='dfi'):
    data_pdb = os.path.join('systems', 'pdb', 'delta_dfi_wt_dL11.pdb')
    err_pdb = os.path.join('systems', 'pdb', 'delta_dfi_wt_dL11_err.pdb')
    atoms = io.pdb2atomlist(data_pdb)
    atoms_data = atoms.mask(['CA', 'P'], mode='name')
    atoms = io.pdb2atomlist(err_pdb)
    atoms_err = atoms.mask(['CA', 'P'], mode='name')
    # atoms_data = atoms_data.mask('Q', mode='chid') # for HPF
    # atoms_err = atoms_err.mask('Q', mode='chid') # for HPF
    in_pdb = 'pdb/obg_dL11.pdb'
    atoms = io.pdb2atomlist(in_pdb)
    pocket = [x-1 for x in atoms.atids]
    # atoms_data = atoms_data.mask(pocket, mode='atid')
    # atoms_err = atoms_err.mask(pocket, mode='atid')
    atoms_data = [atoms_data[x].bfactor for x in pocket]
    atoms_err = [atoms_err[x].bfactor for x in pocket]
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
    y = np.array(atoms_data)
    err = np.array(atoms_err)
    x = np.arange(len(y))
    ax.plot(x, y, label=r'$\Delta L11-WT$', lw=1.5, color='k')
    ax.fill_between(x, y - err, y + err, alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
    ax.set_xlim(0, 106)
    # ax.set_ylim(-0.6, 0.1)
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlabel(r'$Residue$', fontsize=12)
    ax.set_ylabel(r'$\Delta DFI$', fontsize=12)
    ax.set_title('OBG Binding Site', fontsize=12)
    plt.tight_layout()
    fig.savefig(f'png/ddfi_obg_dL11_wt.png')
    plt.close()


if __name__ == '__main__':  
    sysdir = 'systems'
    system_1 = gmxmd.GmxSystem(sysdir, 'ribosome_dL11')
    system_2 = gmxmd.GmxSystem(sysdir, 'ribosome_wt')
    # make_delta_pdb_for_dL11(system_1, system_2, 'gdci_obg', 'ddci_obg_dL11_wt')
    # make_delta_enm_pdb_for_dL11(system_1, system_2, 'dfi_enm', 'dimer_enm_ddfi')
    # wt_pckt_ids = get_atids_from_pdb('rela_wt.pdb')
    # dfi_pdb(system_2, system_1, 'dfi', 'dfi_dL11')
    pocket_ddfi_png(system_1, system_2)
    # arms_ddfi_png(in_pdb='pyscripts/l1_arm_dl11_wt.pdb')