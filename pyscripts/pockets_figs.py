import numpy as np
import MDAnalysis as mda
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from reforge.mdsystem.mdsystem import MDSystem
from reforge import io

def mask(alist, name):
    return np.array([i == name for i in alist])

def filt(arr, bound):
    return arr[arr < bound]

spc = [4215, 4216, 4217, 4218, 4219, 4220, 4221, 4345, 4346, 4347]
ptc_wt = [703, 704, 705, 1800, 1801, 2160, 2161, 2162, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2380, 2381, 2568, 2570, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2681, 2682, 2683, 2684, 2685, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2730, 2731, 2732, 2733, 2737, 2738, 2739, 2740, 2741, 4784, 4785, 4786, 4937, 4938, 4939, 4940, 5375, 8560, 8561, 8562, 8563, 8565, 8569, 8570, 8571, 8572, 8573, 8574, 8575, 8576, 8577, 8578, 8579, 8580]
ptc_dl11 = [5375, 5376, 4937, 4784, 4938, 4785, 4939, 4786, 4940, 8417, 8418, 8421, 8422, 8423, 8424, 8425, 8426, 8427, 8429, 8430, 8431, 8432, 8433, 8434, 8435, 8436, 8437, 8438, 8439, 8440, 8441, 8442, 703, 704, 1259, 1800, 1801, 2160, 2161, 2162, 2163, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2380, 2568, 2570, 2575, 2576, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2619, 2620, 2621, 2622, 2623, 2624, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2681, 2682, 2683, 2684, 2685, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2730, 2731, 2732, 2733, 2737, 2738, 2739, 2740, 2741]
rela_wt = [1011, 1012, 1013, 1014, 1015, 1016, 1022, 1023, 1024, 1025, 1026, 1193, 1194, 1195, 1202, 1203, 1204, 1205, 1207, 1208, 4107, 4108, 4109, 4110, 4111, 4137, 4138, 4882, 4917, 4919, 4920, 4921, 4922, 7840, 7842, 7843, 7844, 7861, 7863, 7866, 8105, 8109, 9653, 9654, 9655, 9656, 9657, 9703, 9704, 9705, 9706]
rela_dl11 = [4882, 7834, 7835, 7836, 7840, 4917, 7842, 4919, 7843, 4920, 7844, 9514, 4921, 9515, 7845, 4922, 9516, 9517, 9518, 7860, 7861, 7862, 7863, 7865, 7866, 9564, 9565, 9566, 9567, 1011, 1012, 1015, 1016, 1022, 1023, 1024, 1026, 4107, 4108, 4109, 4110, 4111, 4137, 4138, 4139, 1193, 1194, 1202, 1205, 1206, 1207, 1209]

# print(np.average(metric_wt), np.std(metric_wt)) 
# print(np.average(metric_mg), np.std(metric_mg)) 
# print(np.average(metric_mgh), np.std(metric_mgh)) 

# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# bins = 25
# ax.hist([metric_wt, metric_mg, metric_mgh], bins=bins, histtype='bar', linewidth=1.5, label=['WT', 'MG', 'MGH'], density=True)
# # ax.hist(metric_mg, bins=bins, histtype='bar', linewidth=1.5, label='MG', density=True)
# # ax.hist(metric_mgh, bins=bins, histtype='bar', linewidth=1.5, label='MGH', density=True)
# # ax.set_xlim(0, 8)
# ax.legend()
# ax.set_xlabel(f'{metric}, A', fontsize=12)
# ax.set_ylabel('Distribution Density', fontsize=12)
# plt.tight_layout()
# fig.savefig(f'png/bb_pocket_{metric}.png')
# plt.close()

sysdir = 'systems'
system_wt = MDSystem(sysdir, 'ribosome_wt')
system_mg = MDSystem(sysdir, 'ribosome_mg')
system_mgh = MDSystem(sysdir, 'ribosome_mgh')

metric = 'dfi'
data_wt = np.load(system_wt.root / 'data' / f'{metric}_av.npy')
data_mg = np.load(system_mg.root / 'data' / f'{metric}_av.npy')
data_mgh = np.load(system_mgh.root / 'data' / f'{metric}_av.npy')
derr_wt = np.load(system_wt.root / 'data' / f'{metric}_err.npy')
derr_mg = np.load(system_mg.root / 'data' / f'{metric}_err.npy')
derr_mgh = np.load(system_mgh.root / 'data' / f'{metric}_err.npy')


# FIRST FIGURE
in_pdb = 'pdb/l1_arm_wt.pdb'
atoms = io.pdb2atomlist(in_pdb)
pocket = [x-1 for x in atoms.atids]

metric_wt = data_wt[pocket] 
metric_mg = data_mg[pocket] 
metric_mgh = data_mgh[pocket] 
err_wt = derr_wt[pocket] 
err_mg = derr_mg[pocket] 
err_mgh = derr_mgh[pocket] 

fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2))
y = metric_mg - metric_wt
err = np.sqrt(err_mg**2 + err_wt**2)
x = np.arange(len(y))
ax.plot(x, y, label='MG-WT', color='#D43F3A')
ax.fill_between(x, y - err, y + err, alpha=0.3, color='#D43F3A')
y = metric_mgh - metric_wt
err = np.sqrt(err_mgh**2 + err_wt**2)
x = np.arange(len(y))
ax.plot(x, y, label='MGH-WT', color='k')
ax.fill_between(x, y - err, y + err, alpha=0.3, color='k')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_ylim(-0.55, 1.05)
ax.set_xlim(0, 270)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.legend(frameon=False, loc='upper right')
ax.set_xlabel(r'$Residue$', fontsize=12)
ax.set_ylabel(r'$\Delta DFI$', fontsize=12)
ax.set_title('L1 Arm', fontsize=14)
plt.tight_layout()
fig.savefig(f'png/ddfi_l1.png')
plt.close()


# SECOND FIGURE
in_pdb = 'pdb/l12_arm_wt.pdb'
atoms = io.pdb2atomlist(in_pdb)
pocket = [x-1 for x in atoms.atids]

metric_wt = data_wt[pocket] 
metric_mg = data_mg[pocket] 
metric_mgh = data_mgh[pocket] 
err_wt = derr_wt[pocket] 
err_mg = derr_mg[pocket] 
err_mgh = derr_mgh[pocket] 

fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2))
ax.plot(x, y, label='MG-WT', color='#D43F3A')
ax.fill_between(x, y - err, y + err, alpha=0.3, color='#D43F3A')
y = metric_mgh - metric_wt
err = np.sqrt(err_mgh**2 + err_wt**2)
x = np.arange(len(y))
ax.plot(x, y, label='MGH-WT', color='k')
ax.fill_between(x, y - err, y + err, alpha=0.3, color='k')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_ylim(-0.55, 1.55)
ax.set_xlim(0, 270)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
# ax.legend(frameon=False, loc='lower right')
ax.set_xlabel(r'$Residue$', fontsize=12)
ax.set_ylabel(r'$\Delta DFI$', fontsize=12)
ax.set_title('L7/12 Arm', fontsize=14)
plt.tight_layout()
fig.savefig(f'png/ddfi_l12.png')
plt.close()