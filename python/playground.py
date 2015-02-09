#!/Users/syt/.virtualenvs/science/bin/python

import numpy as np
from numpy.linalg import eigvals


def run_experiment(niter=100):
    assert 100 < niter
    K = 100
    results = []
    for _ in xrange(niter):
        mat = np.random.randn(K, K)
        max_eigenvalue = np.abs(eigvals(mat)).max()
        results.append(max_eigenvalue)
    return results
some_results = run_experiment(101)
print 'Largest one we saw: {}'.format(np.max(some_results))
