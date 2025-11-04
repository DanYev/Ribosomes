from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import cupy as cp
import cupyx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from reforge import io, mdm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.utils import clean_dir, get_logger

logger = get_logger(__name__)

def calc_eigensystem(sysdir, sysname, n_modes=200, dtype=np.float32, **kwargs):
    kwargs.setdefault("k", n_modes)
    kwargs.setdefault("which", "LM")
    kwargs.setdefault("tol", 0)
    kwargs.setdefault("maxiter", None)
    logger.info(f"Computing {n_modes} normal modes using GPU...")
    system = MDSystem(sysdir, sysname)
    matrix = np.load(system.datdir / "covmat_av.npy")
    matrix_gpu = cp.asarray(matrix, dtype)
    evals_gpu, evecs_gpu = cupyx.scipy.sparse.linalg.eigsh(matrix_gpu, **kwargs)
    logger.info("Normal modes computed.")
    logger.info("Transferring data back to CPU...")
    evals = cp.asnumpy(evals_gpu)
    evecs = cp.asnumpy(evecs_gpu)
    logger.info("Data transfer complete.")
    logger.info("Saving results...")
    np.save("data/evals.npy", evals)
    np.save("data/evecs.npy", evecs)
    return evals, evecs

def calc_nm_from_pdb(sysdir, sysname, n_modes=200, selection="name CA", dtype=np.float32, **kwargs):
    kwargs.setdefault("k", n_modes)
    kwargs.setdefault("which", "SA")
    kwargs.setdefault("tol", 0)
    kwargs.setdefault("maxiter", None) 
    logger.info(f"Loading structure from PDB...")
    system = MDSystem(sysdir, sysname)
    in_pdb = system.inpdb
    u = mda.Universe(in_pdb)
    ag = u.select_atoms(selection)
    vecs = np.array(ag.positions).astype(np.float64) # (n_atoms, 3)
    logger.info("Building Hessian matrix...")
    hess = mdm.hessian(vecs, spring_constant=1.0, cutoff=15, dd=0)
    logger.info(f"Computing {n_modes} normal modes on GPU...")
    matrix_gpu = cp.asarray(hess, dtype)
    evals_gpu, evecs_gpu = cupyx.scipy.sparse.linalg.eigsh(matrix_gpu, **kwargs)
    logger.info("Normal modes computed.")
    logger.info("Transferring data back to CPU...")
    evals = cp.asnumpy(evals_gpu)
    evecs = cp.asnumpy(evecs_gpu)
    inv_evals = evals[6:]**-1 # remove first 6 zero modes
    inv_evecs = evecs[:, 6:] # remove first 6 zero modes
    logger.info("Data transfer complete.")
    logger.info("Saving results...")
    np.save("data/evals.npy", inv_evals)
    np.save("data/evecs.npy", inv_evecs)


def plot_freqs():
    logger.info("Plotting normal mode frequencies...")
    evals = np.load("data/evals.npy")
    factor = 5 / np.pi * np.sqrt(5000)
    freqs = factor / evals
    ys = freqs[::-1]
    xs = np.arange(1, len(freqs) + 1)
    plt.figure()
    plt.plot(xs, ys, marker='o', linestyle='None')
    plt.yscale('log')
    plt.xlabel("Mode index")
    plt.ylabel("Mode Frequency (GHz)")
    plt.title("Normal Modes Spectrum")
    plt.tight_layout()
    plt.savefig("png/frequencies.png", dpi=300)
    plt.close()


def write_mode_trajectory(mode_index, pdb_file, selection="name CA", n_frames=50, n_oscillations=2, 
                          amplitude_scale=1.0, output_file=None):
    """
    Write a trajectory showing the motion along a specific normal mode.
    
    Parameters
    ----------
    mode_index : int
        Index of the mode to visualize (0-indexed)
    pdb_file : str
        Path to the PDB file for topology
    n_frames : int
        Number of frames per oscillation
    n_oscillations : int
        Number of complete oscillations to show
    amplitude_scale : float
        Scaling factor for the eigenvalue-based amplitude (default 1.0)
    output_file : str, optional
        Output XTC file path. If None, uses 'data/mode_{mode_index}.xtc'
    """
    logger.info(f"Creating trajectory for mode {mode_index}...")
    # Load eigenvectors, eigenvalues, and topology
    evecs = np.load("data/evecs.npy")
    evals = np.load("data/evals.npy")
    u = mda.Universe(pdb_file)
    # Get the eigenvector and eigenvalue for this mode
    eigenvec = evecs[:, mode_index]
    eigenvalue = evals[mode_index]
    # Calculate amplitude from eigenvalue (eigenvalue ~ variance ~ amplitude^2)
    # Use sqrt of eigenvalue and scale it
    amplitude = amplitude_scale * np.sqrt(eigenvalue)
    # Adjust this selection based on your system
    atoms = u.select_atoms(selection)
    # Check dimensions
    n_atoms = len(atoms)
    expected_dim = n_atoms * 3
    if len(eigenvec) < expected_dim:
        # Pad eigenvector with zeros to match topology dimension
        logger.warning(f"Eigenvector dimension ({len(eigenvec)}) < expected ({expected_dim})")
        logger.warning(f"Padding with {expected_dim - len(eigenvec)} zeros")
        eigenvec = np.pad(eigenvec, (0, expected_dim - len(eigenvec)), mode='constant', constant_values=0)
    elif len(eigenvec) > expected_dim:
        # Truncate eigenvector if it's too long
        logger.warning(f"Eigenvector dimension ({len(eigenvec)}) > expected ({expected_dim})")
        logger.warning(f"Truncating to {expected_dim}")
        eigenvec = eigenvec[:expected_dim]
    # Reshape eigenvector to (n_atoms, 3) for coordinates
    eigenvec_coords = eigenvec.reshape(n_atoms, 3)
    # Normalize eigenvector for consistent amplitude
    norm = np.linalg.norm(eigenvec_coords)
    eigenvec_coords = eigenvec_coords / norm
    
    # Calculate frequency from eigenvalue
    factor = 5 / np.pi * np.sqrt(5000)  # Same factor as in plot_freqs()
    frequency = factor / eigenvalue  # in GHz
    # Calculate RMSF for this mode (RMS of all atomic displacements)
    rmsf = amplitude * np.sqrt(np.mean(np.sum(eigenvec_coords**2, axis=1)))
    logger.info(f"Mode {mode_index}: eigenvalue = {eigenvalue:.6f}, frequency = {frequency:.4f} GHz, RMSF = {rmsf:.2f} nm")
    
    # Store original positions
    original_positions = atoms.positions.copy()
    # Set output file names
    if output_file is None:
        output_file = f"data/mode_{mode_index}.xtc"
    # Determine PDB output filename
    output_path = Path(output_file)
    pdb_output = output_path.parent / f"{output_path.stem}.pdb"
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save topology as PDB
    atoms.write(str(pdb_output))
    logger.info(f"Topology saved to {pdb_output}")
    # Generate trajectory frames
    total_frames = n_frames * n_oscillations
    with mda.Writer(str(output_file), atoms.n_atoms) as W:
        for frame in range(total_frames):
            # Calculate phase (0 to 2*pi per oscillation)
            phase = 2 * np.pi * frame / n_frames
            # Displacement along mode: amplitude * sin(phase)
            displacement = amplitude * np.sin(phase) * eigenvec_coords
            # Update positions
            atoms.positions = original_positions + displacement
            # Write frame
            W.write(atoms)
    logger.info(f"Trajectory saved to {output_file}")
    logger.info(f"Total frames: {total_frames} ({n_oscillations} oscillations)")
    return str(output_file), str(pdb_output)


if __name__ == "__main__":
    sysdir = "systems" 
    sysname = "ribosome_wt"
    # RUNS
    # calc_eigensystem(sysdir, sysname)
    # plot_freqs()
    selection = "name CA or name P or name C1'"
    # calc_nm_from_pdb(sysdir, sysname, selection=selection, n_modes=20, dtype=np.float32)
    pdb_file = f"{sysdir}/{sysname}/inpdb.pdb"
    for mode_idx in range(3):
        write_mode_trajectory(
            mode_index=mode_idx,
            pdb_file=pdb_file,
            selection=selection,
            n_frames=25,
            n_oscillations=2,
            amplitude_scale=10.0, # 1 for cov matrix, ~10 for hessian-based depending on spring k
        ) 

