"""Publication-style plots from saved D traces; never invent phase trajectories."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def save_figure(figure, path):
    """Export a legible Times New Roman figure in raster and vector formats."""
    figure.savefig(Path(path).with_suffix('.png'), dpi=180, bbox_inches='tight')
    figure.savefig(Path(path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


def plot_study(output):
    """Compare tracking, phase portraits, current commands and internal activity."""
    output = Path(output)
    for directory in (Path.home()/'.local/share/fonts', Path.home()/'.wine/drive_c/windows/Fonts'):
        for path in directory.rglob('times*.ttf'):
            font_manager.fontManager.addfont(str(path))
    family = 'Times New Roman' if any(f.name == 'Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
    plt.rcParams.update({'font.family':family, 'font.size':15,
                         'axes.labelsize':16, 'axes.titlesize':17, 'legend.fontsize':12,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    methods = ('frozen', 'command_only', 'joint')
    labels = ('Frozen C', 'Current-law refinement', 'Joint refinement')
    colors = ('#666666', '#c07716', '#1765ad')
    example = 4  # amplitude 30 degrees, period 20 s; identical in every method.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for name, label, color in zip(methods, labels, colors):
        with np.load(output/name/f'diagnostics/case_{example:03d}.npz') as data:
            tr, ref = data['trace'], data['reference']
            sl = slice(None, None, 50)
            t = tr[sl, 0]
            axes[0, 0].plot(t, np.rad2deg(tr[sl, 1]), color=color, label=label)
            axes[0, 1].plot(t, np.rad2deg(tr[sl, 1]-ref[sl, 0]), color=color)
            mask = (tr[:, 0] >= 6) & (np.arange(len(tr)) % 50 == 0)
            axes[1, 0].plot(np.rad2deg(tr[mask, 1]), tr[mask, 2], color=color, alpha=.8, lw=.8)
            if name == 'frozen':
                axes[0, 0].plot(t, np.rad2deg(ref[sl, 0]), 'k--', lw=1.5, label='Target')
                axes[1, 0].plot(np.rad2deg(ref[mask, 0]), ref[mask, 1], 'k--', lw=1.5, label='Target orbit')
        if name != 'frozen':
            with (output/name/'history.csv').open() as stream:
                history = list(csv.DictReader(stream))
            axes[1, 1].plot([int(r['generation']) for r in history],
                            [float(r['best_fitness']) for r in history], color=color, label=label)
    for ax, title, xlabel, ylabel in zip(axes.ravel(),
            ('A. Angle tracking: amplitude 30°, period 20 s', 'B. Tracking error',
             'C. Phase trajectories after onset', 'D. Training fitness'),
            ('Time (s)', 'Time (s)', 'Angle (deg)', 'Generation'),
            ('Angle (deg)', 'Error (deg)', 'Angular velocity (rad/s)', 'Robust fitness')):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.25)
    axes[0, 0].legend(loc='upper right')
    axes[1, 0].legend()
    axes[1, 1].legend()
    save_figure(fig, folder/'tracking_comparison')

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    matrices = []
    for name in methods:
        with (output/name/'diagnostics/metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        matrices.append(np.rad2deg([float(r['tracking_rmse_rad']) for r in rows]).reshape(3, 3))
    vmax = max(m.max() for m in matrices)
    for ax, label, matrix in zip(axes, labels, matrices):
        artist = ax.imshow(matrix, origin='lower', vmin=0, vmax=vmax, cmap='magma_r', aspect='auto')
        ax.set(xticks=range(3), xticklabels=[10, 20, 40], yticks=range(3), yticklabels=[15, 30, 45],
               xlabel='Period (s)', ylabel='Amplitude (deg)', title=label)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{matrix[i,j]:.2f}°', ha='center', va='center',
                        color='white' if matrix[i,j] > .55*vmax else 'black', fontsize=17)
    fig.colorbar(artist, ax=axes, label='Tracking RMSE (deg)', shrink=.8)
    save_figure(fig, folder/'amplitude_period_rmse')

    selection = json.loads((output/'selection.json').read_text())
    selected = selection['method']
    with np.load(output/selected/f'diagnostics/case_{example:03d}.npz') as data:
        tr, ref, commands = data['trace'], data['reference'], data['commands']
    sl = slice(None, None, 50)
    t = tr[sl, 0]
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), layout='constrained')
    axes[0, 0].plot(t, np.rad2deg(tr[sl, 1]), label='Limb')
    axes[0, 0].plot(t, np.rad2deg(ref[sl, 0]), 'k--', label='Target')
    axes[0, 1].plot(t, commands[sl, :2], label=['Neuron 0', 'Neuron 1'])
    axes[1, 0].plot(t, tr[sl, 3:5], label=['Flexor', 'Extensor'])
    axes[1, 1].plot(t, tr[sl, 11:14].sum(axis=1), label='Flexor Ia+II+Ib')
    axes[1, 1].plot(t, tr[sl, 14:17].sum(axis=1), label='Extensor Ia+II+Ib')
    window = (tr[:, 0] >= 14) & (tr[:, 0] <= 14.3)
    axes[2, 0].plot(tr[window, 0], tr[window, 5:7], lw=.8, label=['Neuron 0', 'Neuron 1'])
    axes[2, 1].plot(t, tr[sl, 17:19], label=['Flexor', 'Extensor'])
    for ax, title, ylabel in zip(axes.ravel(),
            ('A. Limb and target', 'B. Applied neural currents', 'C. Muscle forces',
             'D. Summed afferent responses', 'E. Membrane potentials (full resolution)', 'F. Muscle activation'),
            ('Angle (deg)', 'Current (model units)', 'Force (N)', 'Response (model units)',
             'Membrane potential (mV)', 'Hill activation')):
        ax.set(title=title, xlabel='Time (s)', ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=.25)
    fig.suptitle(f'Validation-selected solution: {selected}; amplitude 30°, period 20 s', fontsize=19)
    save_figure(fig, folder/'state_example')

    fig, axes = plt.subplots(3, 3, figsize=(15, 13), layout='constrained')
    for index, ax in enumerate(axes.ravel()):
        with np.load(output/selected/f'diagnostics/case_{index:03d}.npz') as data:
            tr, ref = data['trace'], data['reference']
        sample = np.arange(0, len(tr), 50)
        warm = tr[sample, 0] < 6
        ax.plot(np.rad2deg(tr[sample[warm], 1]), tr[sample[warm], 2], color='gray', alpha=.6, lw=.7)
        ax.plot(np.rad2deg(tr[sample[~warm], 1]), tr[sample[~warm], 2], lw=.8, label='Limb')
        ax.plot(np.rad2deg(ref[sample[~warm], 0]), ref[sample[~warm], 1], 'k--', lw=1, label='Target')
        ax.set(title=f'A = {(15,30,45)[index//3]}°, T = {(10,20,40)[index%3]} s',
               xlabel='Angle (deg)', ylabel='Angular velocity (rad/s)')
        ax.grid(alpha=.25)
    axes[0, 0].legend()
    save_figure(fig, folder/'phase_trajectories')
