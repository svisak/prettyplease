# Copyright 2021 Isak Svensson

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3

import decimal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

def compute_sigma_levels(sigmas):
    return 1.0 - np.exp(-0.5 * np.array(sigmas) ** 2)

def contour_levels(grid, levels=compute_sigma_levels([1.0, 2.0])):
    """Compute contour levels for a gridded 2D posterior"""
    sorted_ = np.flipud(np.sort(grid.ravel()))
    pct = np.cumsum(sorted_) / np.sum(sorted_)
    cutoffs = np.searchsorted(pct, np.array(levels))
    return np.sort(sorted_[cutoffs])

def corner(data, bins=30, quantiles=[0.16, 0.84], weights=None, **kwargs):
    """
    Create a pretty corner plot.

    :param bins:
        The number of bins to use in both the 1D and 2D histograms.
        Default: 30

    :param levels:
        The levels of the 2D contours showed in the lower triangle.
        Default: [0.68, 0.95]

    :param quantiles:
        The quantiles used to compute uncertainty in the 1D marginalizations.
        Must be of length 2.
        Default: [0.16, 0.84].

    :param weights:
        Array of weights for each sample. Passed to the histogramming functions
        in numpy. Should have shape (len(data),).
        If None, all samples have equal weight.
        Default: None

    :param n_uncertainty_digits:
        Determines to how many significant digits the uncertainty is computed.
        This directly affects how estimates and ticks are displayed.
        Default: 2

    :param labels:
        List of parameter labels. Should be the same length as data.shape[1]
        Default: None

    :param show_estimates:
        Whether to display parameter estimates above the diagonal.
        Uses the median as the central value and uncertainty based
        on the quantiles argument.
        Default: True

    :param plot_estimates:
        Whether to add vertical lines at the quantiles in the 1D plots.
        Default: False

    :param colors:
        Color scheme to use. May be either a single color string
        or a list of colors.
        Default: ['whitesmoke', 'black']

    :param n_ticks:
        (Maximum) number of ticks to show on each axis.
        Default: 4

    :param figsize:
        The figsize. Either a tuple or None.
        Default: None.

    :param fontsize:
        Fontsize to use. Affects all text.
        Default: 10

    :param linewidth:
        Linewidth to use in plots.
        Default: 0.6
    """

    def determine_num_decimals(x, n_uncertainty_digits):
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

    def grow_x(ax):
        ticks = ax.get_xticks()
        lim = ax.get_xlim()
        low, high, changed = new_limits(ticks, lim, 7)
        if changed:
            ax.set_xlim(low, high)

    def grow_y(axes, row):
        ax = axes[row, 0]
        ticks = ax.get_yticks()
        lim = ax.get_ylim()
        low, high, changed = new_limits(ticks, lim, 8)
        if changed:
            for ax in axes[row, :row]:
                ax.set_ylim(low, high)

    def new_limits(ticks, limits, tick_free_zone_factor):
        low = limits[0]
        high = limits[1]
        r = high - low
        red_zone = r / tick_free_zone_factor
        ticks = [t for t in ticks if t >= low and t <= high]
        changed = False
        if ticks[0] < low + red_zone or ticks[-1] > high - red_zone:
            low = low - red_zone
            high = high + red_zone
            changed = True
        return (low, high, changed)

    # The length of quantiles must be 2
    assert(len(quantiles) == 2)

    # Pop keyword arguments
    levels = kwargs.pop('levels', None)
    n_uncertainty_digits = kwargs.pop('n_uncertainty_digits', 2)
    labels = kwargs.pop('labels', None)
    plot_estimates = kwargs.pop('plot_estimates', False) # Show vertical lines at quantiles?
    show_estimates = kwargs.pop('show_estimates', True) # Show median and uncertainty above diagonal
    colors = kwargs.pop('colors', ['whitesmoke', 'black'])
    n_ticks = kwargs.pop('n_ticks', 4)
    xticklabel_rotation = kwargs.pop('xticklabel_rotation', 45)
    figsize = kwargs.pop('figsize', None)
    fontsize = kwargs.pop('fontsize', 10)
    lw = kwargs.pop('linewidth', 0.7)

    # Default levels
    if levels is None:
        levels = compute_sigma_levels([1.0, 2.0])

    # Color scheme. If colors is a string then the color scheme is
    # "white plus the specified color".
    # If colors is a list the user has completely
    # specified the color scheme they want.
    if type(colors) is str:
        colors = ['whitesmoke', colors]

    density_cmap = LinearSegmentedColormap.from_list("density_cmap", colors=colors)
    ndim = data.shape[1]

    # Determine suitable number of digits to show
    decimals = [determine_num_decimals(column, n_uncertainty_digits) for column in data.T]

    # n_ticks can be either an int or a list
    if type(n_ticks) is int:
        n_ticks = [n_ticks] * ndim
    assert(len(n_ticks) == ndim)

    if figsize is None:
        # TODO This autoscaling is very crude
        tmp = 5 + ndim
        figsize = (tmp, tmp)
    tmp = {}
    tmp['squeeze'] = False
    tmp['sharex'] = 'col' # We cannot sharey because of the ylim on the diagonal
    tmp['figsize'] = figsize
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, **tmp)

    # Upper triangle
    for col in range(ndim):
        for row in range(col):
            axes[row, col].axis('off')
    # Diagonal
    for i in range(ndim):
        x = data[:, i].flatten()
        ax = axes[i, i]
        ax.hist(x, bins=bins, color=colors[-1], histtype='step', linewidth=lw, density=True, weights=weights)
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
            ax.set_title(title, fontsize=fontsize, loc='center')
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
            hist, xedges, yedges = np.histogram2d(x1, x2, bins=bins, weights=weights)
            hist = hist.T
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            n_contourf_levels = 30
            locator = ticker.MaxNLocator(n_contourf_levels, min_n_ticks=1)
            lev = locator.tick_values(np.min(hist), np.max(hist))
            lev[0] = lev[0] if lev[0] > 0 else 1
            ax.contourf(hist, extent=extent, cmap=density_cmap, levels=lev, extend='max')
            tmp = contour_levels(hist, levels)
            ax.contour(hist, extent=extent, colors='xkcd:charcoal gray', linewidths=lw, levels=tmp, alpha=0.5)

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
    for i in range(ndim):
        ax = axes[-1,i]
        locator = ticker.MaxNLocator(n_ticks[i])
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatters[i])
        grow_x(ax)
        [l.set_fontsize(fontsize) for l in ax.get_xticklabels()]
        [l.set_rotation(xticklabel_rotation) for l in ax.get_xticklabels()]
        [l.set_horizontalalignment('right') for l in ax.get_xticklabels()]
    for i in range(1,ndim):
        ax = axes[i,0]
        locator = ticker.MaxNLocator(n_ticks[i])
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatters[i])
        grow_y(axes, i)
        [l.set_fontsize(fontsize) for l in ax.get_yticklabels()]

    # Adjust plot
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig
