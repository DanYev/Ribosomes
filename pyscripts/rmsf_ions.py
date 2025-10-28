import numpy as np
import pickle
import matplotlib.pyplot as plt
import MDAnalysis as mda
from pathlib import Path
from scipy.stats import bootstrap
from reforge.mdsystem.mdsystem import MDSystem
from reforge.io import pull_files

def mask(alist, name):
    return np.array([i == name for i in alist])


def filt(arr, bound):
    return arr[arr < bound]


def make_masks(system):
    u = mda.Universe(system.root / 'mdci.pdb', in_memory=True)
    atoms = u.atoms
    k_mask = mask(atoms.names, 'K')
    mg_mask = mask(atoms.names, 'MG')
    mgh_mask = mask(atoms.names, 'MGH')
    p_mask =  mask(atoms.names, 'BB1')
    ca_mask =  mask(atoms.names, 'BB')
    masks = k_mask, mg_mask, mgh_mask, p_mask, ca_mask
    # masks = p_mask, ca_mask
    return masks


def get_dist(fpath, masks):
    data = np.load(fpath).astype(np.float32)
    datas = []
    for mask in masks:
        datas.append(data[mask] * 10)
    return tuple(datas)


def get_dists(system, files):
    masks = make_masks(system)
    datas = [[] for _ in masks]
    for f in files:
        dists = get_dist(f, masks)
        for data, dist in zip(datas, dists):
            data.append(dist)
    datas = [filt(np.array(data).flatten(), 9) for data in datas]
    for data in datas:
        print(np.average(data), np.std(data))  
    return datas


def calc_hists(data, n_bins, hist_range):

    def hist_counts(data):
        counts, _ = np.histogram(data, bins=n_bins, range=hist_range)
        return counts

    res = bootstrap(
        data=(data, ),
        statistic=hist_counts,
        vectorized=False,     
        n_resamples=100,     
        method='BCa'   
    )
    ci_low  = res.confidence_interval.low
    ci_high = res.confidence_interval.high
    counts, edges = np.histogram(data, bins=n_bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, ci_low, ci_high


def plot_hist_per_var():    
    sysdir = 'systems'
    sysname = f'ribosome_{var}' 
    system = MDSystem(sysdir, sysname)
    files = pull_files(system.mddir, 'rmsf.npy')
    datas = get_dists(system, files)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    colors = ["xkcd:cerulean", "xkcd:orange", "xkcd:jungle green", "xkcd:black", "xkcd:silver"]
    markers = ['o-', '^-', 's-', 'v-', 'v-']
    labels = ['k', 'mg', 'mgh', ] # 'p', 'ca']
    for (i, data), color, label, marker in zip(enumerate(datas), colors, labels, markers):
        if data.size != 0:
            centers, counts, ci_low, ci_high = calc_hists(data, n_bins=16, hist_range=(0.5, 8.5))
            # ax.hist(data, bins=bins, histtype='bar', linewidth=1.5, density=True) # label=['k', 'mg', 'mgh'],
            ax.errorbar(centers, counts / np.sum(counts), 
                yerr=[(counts - ci_low) / np.sum(counts), (ci_high - counts) / np.sum(counts) ], 
                fmt=marker, capsize=3, color=color, label=label)
            picl = centers, counts, ci_low, ci_high
            with open(f'data/hist_{label}_{var}.pkl', 'wb') as f:
                pickle.dump(picl, f)
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 0.55)
    ax.legend()
    ax.set_xlabel(r'RMSF $(\AA)$', fontsize=12)
    ax.set_ylabel(r'Distribution Density ± 95% CI', fontsize=12)
    plt.tight_layout()
    fig.savefig(f'png/ions_{var}_rmsf.png')
    plt.close()


def plot_hist_per_atom(fpaths, outpath='png/hist.png'):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    colors = ["xkcd:cerulean", "xkcd:orange", "xkcd:jungle green", "xkcd:black", "xkcd:silver"]
    markers = ['o-', '^-', 's-', 'v-', 'v-']
    labels = ['wt', 'mg', 'mgh', ]
    for fpath, color, label, marker in zip(fpaths, colors, labels, markers):
        with open(fpath, 'rb') as f:
            centers, counts, ci_low, ci_high = pickle.load(f)
        ax.errorbar(centers, counts / np.sum(counts), 
            yerr=[(counts - ci_low) / np.sum(counts), (ci_high - counts) / np.sum(counts) ], 
            fmt=marker, capsize=3, color=color, label=label)
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 0.55)
    ax.legend()
    ax.set_xlabel(r'RMSF $(\AA)$', fontsize=12)
    ax.set_ylabel(r'Distribution Density ± 95% CI', fontsize=12)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()


if __name__ == "__main__":
    rib_vars = ['wt', 'mg', 'mgh']
    for var in rib_vars:
        plot_hist_per_var()
    # atom_names = ['mg', 'mgh', 'p', 'ca']
    # pickle_files = list(Path('data').glob('hist_*.pkl'))
    # fpathss = [[f for f in pickle_files if f.name.split("_")[1] == aname] for aname in atom_names]
    # for fpaths, label in zip(fpathss, atom_names):
    #     plot_hist_per_atom(fpaths, outpath=f'png/hist_{label}.png')





