#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

def contour_levels(grid, quantiles=[0.68, 0.95, 0.997]):
    """Compute contour levels for a gridded 2D posterior"""
    sorted_ = np.flipud(np.sort(grid.ravel()))
    pct = np.cumsum(sorted_) / np.sum(sorted_)
    cutoffs = np.searchsorted(pct, np.array(quantiles) ** 2)
    return np.sort(sorted_[cutoffs])

def corner(data, bins=50, quantiles=[0.68, 0.95, 0.997], labels=None, colors='blue', title=None, show_estimates=True, n_ticks=4, fmt='.3f', figsize=(10,10)):
    """Create a pretty corner plot."""
    # Color scheme. If colors is a string then the color scheme is white and the specified color.
    # If colors is a list the user has completely specified the color scheme they want.
    if type(colors) is str:
        colors = ['white', colors]

    density_cmap = LinearSegmentedColormap.from_list("density_cmap", colors=colors)
    ndim = data.shape[1]

    tmp = 5 + 1.25 * ndim # TODO Unsure how well this scaling works! Seems fine with 2x2 and 10x10 grids ...
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, squeeze=False, figsize=(tmp,tmp))

    # Upper triangle
    for col in range(ndim):
        for row in range(col):
            axes[row, col].axis('off')
    # Diagonal
    for i in range(ndim):
        x = data[:, i].flatten()
        ax = axes[i, i]
        ax.hist(x, bins=bins, color=colors[-1], histtype='step', linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_estimates:
            median = np.percentile(x, 50)
            low = np.percentile(x, 16) - median
            high = np.percentile(x, 84) - median
            label = f'{labels[i]}\n' if labels is not None else ''
            ax.set_title(f'{label}' rf'${median:{fmt}}_{{{low:{fmt}}}}^{{+{high:{fmt}}}}$')
    # Lower triangle
    for col in range(ndim):
        for row in range(col+1, ndim):
            # DO THIS FOR THE LOWER TRIANGLE
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            x1 = data[:, col]
            x2 = data[:, row]
            hist, xedges, yedges = np.histogram2d(x1, x2, bins=bins)
            hist = hist.T
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            vmin = (np.max(hist)-np.min(hist)) / 50 + np.min(hist) # Make low levels white
            ax.contourf(hist, extent=extent, cmap=density_cmap, levels=30, vmin=vmin)
            contourlevels = contour_levels(hist, quantiles)
            ax.contour(hist, extent=extent, colors='gray', linewidths=0.8, levels=contourlevels, alpha=1.0)

    # Bottom labels
    for col in range(ndim):
        if labels is not None:
            axes[-1, col].set_xlabel(labels[col])
    # Left labels
    for row in range(1,ndim):
        if labels is not None:
            axes[row, 0].set_ylabel(labels[row])

    # Ticks
    for i in range(ndim):
        ax = axes[-1,i]
        locator = ticker.MaxNLocator(n_ticks)
        formatter = ticker.FormatStrFormatter(rf'$%{fmt}$')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        [l.set_horizontalalignment('right') for l in ax.get_xticklabels()]
        #[l.set_fontsize('x-small') for l in ax.get_xticklabels()]
    for i in range(1,ndim):
        ax = axes[i,0]
        locator = ticker.MaxNLocator(n_ticks)
        formatter = ticker.FormatStrFormatter(rf'$%{fmt}$')
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        #[l.set_fontsize('x-small') for l in ax.get_yticklabels()]

    # Adjust plot
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def compare_marginals(chain_list, bins=100, alpha=0.8, chain_labels=None, parameter_labels=None, vlines=None):
    """Compare marginalized distributions from several MCMC samplings."""
    ndim = chain_list[0].shape[1]
    base_size = 4
    nrows = None
    ncols = None
    if ndim <= 3:
        nrows = 1
        ncols = ndim
    else:
        tmp = int(np.ceil(np.sqrt(ndim)))
        nrows = tmp
        ncols = tmp
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(base_size*nrows, base_size*ncols))
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            for j in range(len(chain_list)):
                chain = chain_list[j]
                ax.hist(chain[:, i], bins=bins, density=True, alpha=alpha, label=chain_labels[j])
            if parameter_labels is not None:
                ax.set_xlabel(parameter_labels[i])
            if vlines is not None:
                ax.axvline(vlines[i], color='green', linestyle='--')
            ax.set_yticks([])
            ax.legend()
            i += 1
    return fig

def traceplot(chains, labels=None, figwidth=7):
    if type(chains) is np.ndarray:
        chains = [chains]
    ndim = chains[0].shape[1]
    figheight = 5
    fig = plt.figure(figsize=(figwidth, ndim*figheight))
    for i in range(ndim):
        ax = fig.add_subplot(ndim, 1, i+1)
        for chain in chains:
            ax.plot(chain[:, i])
        if labels is not None:
            ax.set_title(labels[i])
    return fig
