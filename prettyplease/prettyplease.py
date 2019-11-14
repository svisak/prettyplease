#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def corner(data, bins=50, figsize=(10,10)):
    """Create a pretty corner plot."""
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", ['k', (1, 1, 1, 0)])
    ndim = data.shape[1]
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, squeeze=False, figsize=figsize)
    for col in range(ndim):
        for row in range(col):
            # DO THIS FOR THE UPPER TRIANGLE
            axes[row, col].axis('off')
        #for row in range(ndim):
        #    # DO THIS FOR ALL AXES
        #    axes[row, col].axis('square')
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
        axes[col, col].hist(data[:, col], bins=bins, color='black', histtype='stepfilled')
        axes[col, col].set_xticks([])
        axes[col, col].set_yticks([])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def flattened_diagonal(chain_list, bins=100, alpha=0.8, chain_labels=None, parameter_labels=None, vlines=None):
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
    fig.tight_layout()
    return fig

def marginalize_mcmc(data, index=0, bins=100, xrange=None, normalize=True):
    if type(data) is not np.ndarray:
        raise ValueError('data is not an ndarray')
    histdd, edges = np.histogramdd(data, bins=bins, range=xrange, density=normalize)
    bin_width = edges[index][1] - edges[index][0]
    marginalized_data = np.empty(bins)
    marginalize_over = list(range(data.shape[1]))
    marginalize_over.remove(index)
    marginalized_data = np.apply_over_axes(np.sum, histdd, marginalize_over).flat

    if normalize is True:
        marginalized_data = normalize_plot(marginalized_data, bin_width)
    midpoints = edges[index][0:-1] + bin_width / 2
    return (midpoints, marginalized_data)

def normalize_plot(array, bin_width):
    return array / np.sum(array) / bin_width

def marginal_plots(data, indices=None, bins=100, normalize=True):
    """
    Marginalize and plot the input data with respect to the columns
    indicated in indices (a list). If indices is None, plot every index.
    Return the resulting figure.
    """
    if indices is None:
        indices = [i for i in range(data.shape[1])]
    n_indices = len(indices)

    # THE LAYOUT COULD BE IMPROVED ...
    figsize = (5*n_indices, 5)
    fig = plt.figure(figsize=figsize)
    for i in range(len(indices)):
        index = indices[i]
        ax = fig.add_subplot(1, n_indices, i+1)
        x, y = marginalize_mcmc(data, index, bins=bins, normalize=normalize)
        ax.plot(x, y, drawstyle='steps-mid')
    return fig
