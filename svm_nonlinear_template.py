# %% [markdown]
# # Non-linear support vector machine

# %%
# Importing with custom names to avoid issues with numpy / sympy matrix
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# %% [markdown]
# Kernel function

# %%
def kernel_poly2(x1, x2):
    # Calculate the kernel value
    r = (x1.T @ x2 + 1)**2
    return r

# %% [markdown]
# Data onboarding

# %% [markdown]
# Optimisation arguments and optimisation by quadratic programming

# %%
LB = ... # lower bound
C = ... # upper bound

# Specify P, q, G, h, A, b
P = cvxopt_matrix(...)
q = cvxopt_matrix(...)
G = cvxopt_matrix(...)
h = cvxopt_matrix(...)
A = cvxopt_matrix(...)
b = cvxopt_matrix(...)

# Setting solver parameters (change default to decrease tolerance) 
cvxopt_solvers.options['show_progress'] = False
#cvxopt_solvers.options['abstol'] = 1e-10
#cvxopt_solvers.options['reltol'] = 1e-10
#cvxopt_solvers.options['feastol'] = 1e-10

# Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

# %% [markdown]
# Support vectors based on close-to-zero values and discriminant function parameters

# %%
# The support vectors
...

# Solve w and w0
...

# %% [markdown]
# Classification of test data

# %%
# Classification for each test data point
...

# Calculate classification accuracy with the training set
...

# %% [markdown]
# Option: Visualise the results


