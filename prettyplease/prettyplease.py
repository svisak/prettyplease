#!/usr/bin/env python3

import decimal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

def contour_levels(grid, levels=[0.68, 0.95]):
    """Compute contour levels for a gridded 2D posterior"""
    sorted_ = np.flipud(np.sort(grid.ravel()))
    pct = np.cumsum(sorted_) / np.sum(sorted_)
    cutoffs = np.searchsorted(pct, np.array(levels))
    return np.sort(sorted_[cutoffs])

def corner(data, bins=50, levels=[0.68, 0.95], quantiles=[0.16, 0.84], **kwargs):
    """Create a pretty corner plot."""

    def determine_num_decimals(x, n_uncertainty_digits=2):
        n_extra_digits = n_uncertainty_digits - 1
        median = np.percentile(x, 50)
        low = np.percentile(x, 16) - median
        high = np.percentile(x, 84) - median
        decimals_low = -decimal.Decimal(low).adjusted() + n_extra_digits
        decimals_high = -decimal.Decimal(high).adjusted() + n_extra_digits
        decimals = max(decimals_low, decimals_high)
        return max(0, decimals)

    def diagonal_title(label, mid, low, high, n_dec):
        fmt = f'.{n_dec}f'
        mid = f'{mid:{fmt}}'
        low = f'{low:{fmt}}'
        high = f'+{high:{fmt}}'
        label = label + '\n' if label is not None else ''
        return label + rf'${mid}_{{{low}}}^{{{high}}}$'

    # The length of quantiles must be 2
    assert(len(quantiles) == 2)

    # Pop keyword arguments
    n_uncertainty_digits = kwargs.pop('n_uncertainty_digits', 2)
    labels = kwargs.pop('labels', None)
    plot_estimates = kwargs.pop('plot_estimates', False) # Show vertical lines at quantiles?
    show_estimates = kwargs.pop('show_estimates', True) # Show median and uncertainty above diagonal
    colors = kwargs.pop('colors', ['white', 'black'])
    n_ticks = kwargs.pop('n_ticks', 4)
    figsize = kwargs.pop('figsize', None)
    fontsize = kwargs.pop('fontsize', 10)
    lw = kwargs.pop('linewidth', 0.6)

    # Color scheme. If colors is a string then the color scheme is
    # "white plus the specified color".
    # If colors is a list the user has completely
    # specified the color scheme they want.
    if type(colors) is str:
        colors = ['white', colors]

    density_cmap = LinearSegmentedColormap.from_list("density_cmap", colors=colors)
    ndim = data.shape[1]

    # Determine suitable number of digits to show
    decimals = [determine_num_decimals(column, n_uncertainty_digits) for column in data.T]

    if figsize is None:
        # TODO This autoscaling is very crude
        tmp = 5 + ndim
        figsize = (tmp, tmp)
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, squeeze=False, figsize=figsize)

    # Upper triangle
    for col in range(ndim):
        for row in range(col):
            axes[row, col].axis('off')
    # Diagonal
    for i in range(ndim):
        x = data[:, i].flatten()
        ax = axes[i, i]
        ax.hist(x, bins=bins, color=colors[-1], histtype='step', linewidth=lw)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_estimates:
            n_dec = decimals[i]
            median = np.percentile(x, 50)
            tmp = [np.quantile(x, q)-median for q in quantiles]
            low = tmp[0]
            high = tmp[1]
            label = labels[i] if labels is not None else ''
            title = diagonal_title(label, median, low, high, n_dec)
            ax.set_title(title, fontsize=fontsize)
        if plot_estimates:
            c = colors[-1]
            [ax.axvline(np.quantile(x, q), ls='-.', color=c, lw=lw) for q in quantiles]
    # Lower triangle
    for col in range(ndim):
        for row in range(col+1, ndim):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            x1 = data[:, col]
            x2 = data[:, row]
            hist, xedges, yedges = np.histogram2d(x1, x2, bins=bins)
            hist = hist.T
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            vmin = (np.max(hist)-np.min(hist)) / 50 + np.min(hist) # Make low levels white
            ax.contourf(hist, extent=extent, cmap=density_cmap, levels=30, vmin=vmin, extend='max')
            tmp = contour_levels(hist, levels)
            ax.contour(hist, extent=extent, colors=colors[-1], linewidths=lw, levels=tmp, alpha=0.5)

    # Bottom labels
    for col in range(ndim):
        if labels is not None:
            axes[-1, col].set_xlabel(labels[col], fontsize=fontsize)
    # Left labels
    for row in range(1,ndim):
        if labels is not None:
            axes[row, 0].set_ylabel(labels[row], fontsize=fontsize)

    # Ticks
    formatters = [ticker.FormatStrFormatter(rf'$%.{dec}f$') for dec in decimals]
    locator = ticker.MaxNLocator(n_ticks)
    for i in range(ndim):
        ax = axes[-1,i]
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatters[i])
        [l.set_fontsize(fontsize) for l in ax.get_xticklabels()]
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        [l.set_horizontalalignment('right') for l in ax.get_xticklabels()]
    for i in range(1,ndim):
        ax = axes[i,0]
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatters[i])
        [l.set_fontsize(fontsize) for l in ax.get_yticklabels()]

    # Adjust plot
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig
