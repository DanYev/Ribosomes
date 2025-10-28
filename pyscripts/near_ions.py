import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

def pdfi_hists():
    structure = mda.Universe('systems/ribosome_wt/png/dfi.pdb', in_memory=True)
    ions = mda.Universe('systems/ribosome_wt/ions/ions.pdb', in_memory=True)
    u = mda.Merge(structure.atoms, ions.atoms)
    bb = u.select_atoms("name CA or name P")
    k_ions = u.select_atoms("name K")
    mg_ions = u.select_atoms("name MG")
    mgh_ions = u.select_atoms("name MGH")
    ag_k = u.select_atoms("(around 15 (name K)) and (name P or name CA)")
    ag_mg = u.select_atoms("(around 15 (name MG)) and (name P or name CA)")
    ag_mgh = u.select_atoms("(around 15 (name MGH)) and (name P or name CA)")
    dfi_k = ag_k.tempfactors
    dfi_mg = ag_mg.tempfactors
    dfi_mgh = ag_mgh.tempfactors
    print(np.average(dfi_k), np.std(dfi_k)) 
    print(np.average(dfi_mg), np.std(dfi_mg)) 
    print(np.average(dfi_mgh), np.std(dfi_mgh)) 
    fig, ax = plt.subplots(1, 1)
    bins = 25
    ax.hist(dfi_k, bins=bins, histtype='bar', lw=2, label='k', density=True)
    ax.hist(dfi_mg, bins=bins, histtype='step', lw=2, label='mg', density=True)
    ax.hist(dfi_mgh, bins=bins, histtype='step', lw=2, label='mgh', density=True)
    ax.set_xlabel('DFI')
    ax.set_ylabel('Density')
    ax.legend()
    fig.savefig('png/p_dfi.png')
    plt.close()


def get_min_dist(atom, group):
    atom_pos = atom.position
    group_pos = group.positions
    distances = distance_array(atom_pos, group_pos)
    return np.min(distances)


def scatter_main():
    data = mda.Universe('systems/pdb/ddfi1_mg_wt.pdb', in_memory=True)
    err = mda.Universe('systems/pdb/ddfi1_mg_wt_err.pdb', in_memory=True)
    ions = mda.Universe('systems/ribosome_wt/ions/ions.pdb', in_memory=True)
    pocket_pdbs = ['pdb/l1_arm_wt.pdb',
                   'pdb/l12_arm_wt.pdb']
    # Load multiple PDBs and merge them
    pocket_u1 = mda.Universe(pocket_pdbs[0], in_memory=True)
    pocket_u2 = mda.Universe(pocket_pdbs[1], in_memory=True)
    pocket_u = mda.Merge(pocket_u1.atoms, pocket_u2.atoms)
    pocket_ids = pocket_u.atoms.ids
    u = mda.Merge(data.atoms, ions.atoms)
    ag = u.select_atoms("name CA or name P or name K or name MG or name MGH")
    bb = ag.select_atoms("name CA or name P")
    ions = ag.select_atoms("name K or name MG or name MGH")
    bb_err = err.select_atoms("name CA or name P")
    min_dists = [get_min_dist(atom, ions) for atom in bb]
    com_ions = ions.center_of_mass()
    com_dists = [np.linalg.norm(position - com_ions) for position in bb.positions]
    dists = np.array(min_dists)
    datas = np.array(bb.tempfactors)
    errs = np.array(bb_err.tempfactors)
    mask = (np.abs(datas) > errs) 
    datas_plt = datas[mask]
    dists_plt = dists[mask]
    datas_pockets = datas[pocket_ids]
    dists_pockets = dists[pocket_ids]
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8))
    ax.scatter(dists_plt, datas_plt, alpha=0.25, color='#D43F3A')
    ax.scatter(dists_pockets, datas_pockets, alpha=0.1, color='black', marker='o')
    # ax.axhline(y=0.0, color='k', linestyle='-', linewidth=1)
    ax.set_xlim(0, 65)
    ax.set_ylim(-1, 1)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)', fontsize=12)
    ax.set_ylabel(r'$\Delta DFI$', fontsize=12)
    ax.legend(frameon=False)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig('png/ddfi_scatter_dist.png')
    plt.close()


def violin_main():
    data = mda.Universe('systems/pdb/ddfi1_mg_wt.pdb', in_memory=True)
    err = mda.Universe('systems/pdb/ddfi1_mg_wt_err.pdb', in_memory=True)
    ions = mda.Universe('systems/ribosome_wt/ions/ions.pdb', in_memory=True)
    u = mda.Merge(data.atoms, ions.atoms)
    ag = u.select_atoms("name CA or name P or name K or name MG or name MGH")
    bb = ag.select_atoms("name CA or name P")
    ions = ag.select_atoms("name K or name MG or name MGH")
    bb_err = err.select_atoms("name CA or name P")
    min_dists = [get_min_dist(atom, ions) for atom in bb]
    com_ions = ions.center_of_mass()
    com_dists = [np.linalg.norm(position - com_ions) for position in bb.positions]
    dists = np.array(min_dists)
    dists = np.array(min_dists)
    datas = np.array(bb.tempfactors)
    errs = np.array(bb_err.tempfactors)
    mask = np.abs(datas) > errs
    datas = datas[mask]
    dists = dists[mask]
    gr_1 = datas[dists < 10]
    gr_2 = datas[(dists > 10) & (dists < 20)]
    gr_3 = datas[(dists > 20) & (dists < 30)]
    gr_4 = datas[dists > 30]
    groups = [gr_1, gr_2, gr_3, gr_4]
    widths = [2 * len(x) / len(datas) for x in groups]
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8))
    vp = ax.violinplot(groups, widths=widths, showmeans=True, showmedians=False, showextrema=True)
    vp['cmeans'].set_color('k')
    vp['cmeans'].set_linewidth(1) 
    vp['cbars'].set_color('k')
    vp['cbars'].set_linewidth(1)
    for pc in vp['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([r'$< 10 \mathrm{\AA}$', r'$10 - 20 \mathrm{\AA}$', r'$20 - 30 \mathrm{\AA}$', r'$ > 30 \mathrm{\AA}$'], 
        fontsize=12)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax.set_ylabel(r'$\Delta DFI$', fontsize=12)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig('png/ddfi_violin.png')
    plt.close()



if __name__ == "__main__":
    scatter_main()
    # violin_main()
    