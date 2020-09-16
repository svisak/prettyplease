#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def contour_levels(grid, percentiles=[0.68, 0.95, 0.997]):
    """Compute contour levels for a gridded 2D posterior"""
    sorted_ = np.flipud(np.sort(grid.ravel()))
    pct = np.cumsum(sorted_) / np.sum(sorted_)
    cutoffs = np.searchsorted(pct, np.array(percentiles) ** 2)
    return np.sort(sorted_[cutoffs])

def corner(data, bins=50, percentiles=[0.68, 0.95, 0.997], labels=None, title=None, show_estimates=True, fmt='.3f', grayscale=False, figsize=(10,10)):
    """Create a pretty corner plot."""
    #density_cmap = LinearSegmentedColormap.from_list("density_cmap", ['k', (1, 1, 1, 0)])
    colors = ['white', 'gray', 'black'] if grayscale else ['white', 'blue','purple']
    density_cmap = LinearSegmentedColormap.from_list("density_cmap", colors=colors)
    ndim = data.shape[1]
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, squeeze=False, figsize=figsize)

    # Upper triangle
    for col in range(ndim):
        for row in range(col):
            axes[row, col].axis('off')
    # Lower triangle
    for col in range(ndim):
        for row in range(col+1, ndim):
            # DO THIS FOR THE LOWER TRIANGLE
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            x1 = data[:, col]
            x2 = data[:, row]
            hist = np.histogram2d(x1, x2, bins=bins)
            h = hist[0].T
            vmin = (np.max(h)-np.min(h)) / 50 + np.min(h) # Make low levels white
            ax.contourf(h, cmap=density_cmap, levels=30, vmin=vmin)
            contourlevels = contour_levels(hist[0].T, percentiles)
            ax.contour(h, colors='gray', linewidths=0.8, levels=contourlevels, alpha=1.0)
    # Diagonal
    for i in range(ndim):
        x = data[:, i].flatten()
        ax = axes[i, i]
        ax.hist(x, bins=bins, color='black', histtype='step')
        ax.set_xticks([])
        ax.set_yticks([])
        if show_estimates:
            median = np.percentile(x, 50)
            low = np.percentile(x, 16) - median
            high = np.percentile(x, 84) - median
            ax.set_title(rf'${median:{fmt}}_{{{low:{fmt}}}}^{{+{high:{fmt}}}}$')
    # Bottom
    for col in range(ndim):
        if labels is not None:
            axes[-1, col].set_xlabel(labels[col])
    # Left
    for row in range(ndim):
        if labels is not None:
            axes[row, 0].set_ylabel(labels[row])
    # Remove space between subplots
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
