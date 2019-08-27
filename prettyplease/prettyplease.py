#!/usr/bin/env python

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
