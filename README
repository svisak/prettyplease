Create eye-pleasing cornerplots (and maybe other types of plots in the future.)

To install, clone the repository, and run 

  pip install .

Usage example(s):

  import prettyplease
  import matplotlib.pyplot as plt
  import numpy as np
  
  # Generate some example data
  rng = np.random.default_rng()
  n_samples = 100000
  n_dim = 3
  x = rng.normal(1.618, 0.1, size=(n_samples, n_dim))
  # x is an ndarray of shape (n_samples, n_dim)
  
  # With default settings
  fig = prettyplease.corner(x)
  plt.savefig('default.pdf', bbox_inches='tight')
  
  # With some customization
  labels = [rf'$\theta_{i}$' for i in range(n_dim)]
  fig = prettyplease.corner(x, n_uncertainty_digits=4, labels=labels, plot_estimates=True, colors='red')
  plt.savefig('custom.pdf', bbox_inches='tight')
