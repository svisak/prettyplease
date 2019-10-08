#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def corner(data):
    """Create a pretty corner plot."""
    ndim = data.shape[1]
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, squeeze=False)
    for col in range(ndim):
        axes[col, col].hist(data[:, col])
        for row in range(col+1, ndim):
            axes[row, col].hist2d(data[:, col], data[:, row])
        for row in range(col):
            axes[row, col].axis('off')

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

#def marginalize_scan(

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
