#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def corner(data, bins=50, labels=None, title=None, show_estimates=True, figsize=(10,10)):
    """Create a pretty corner plot."""
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", ['k', (1, 1, 1, 0)])
    ndim = data.shape[1]
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, squeeze=False, figsize=figsize)
    for col in range(ndim):
        for row in range(col):
            # DO THIS FOR THE UPPER TRIANGLE
            axes[row, col].axis('off')
        for row in range(col+1, ndim):
            # DO THIS FOR THE LOWER TRIANGLE
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            x1 = data[:, col]
            x2 = data[:, row]
            hist = np.histogram2d(x1, x2, bins=bins)
            ax.imshow(hist[0].T, origin='lower', cmap='binary')
            #ax.contourf(hist[0].T, cmap='gist_yarg', levels=3, alpha=0.3)
            ax.contour(hist[0].T, colors=['gray','black'], linewidths=0.8, levels=3, alpha=1.0)
    for i in range(ndim):
        # THE DIAGONAL
        ax = axes[i, i]
        ax.hist(data[:, i], bins=bins, color='black', histtype='stepfilled')
        ax.set_xticks([])
        ax.set_yticks([])
        if show_estimates:
            median = data[:, i].median()
            ax.set_title(f'{median:.2f}')
    for i in range(ndim):
        # ADD LABELS
        if labels is not None:
            axes[-1, i].set_xlabel(labels[i])
            axes[i, 0].set_ylabel(labels[i])
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
