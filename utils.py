import os
import re
import time

import jax
import jax.numpy as jnp


from collections import namedtuple

from jax.random import PRNGKey


VIResults = namedtuple(
    "VIResults",
    [
        "samples",
        "vardist",
        "var_params",
        'model', 
        'recipe',
        'total_time',
        'time_per_iteration',
        'final_elbo',
        'best_step', 
        'callback_list',

    ]
)
TFPMCMCResults = namedtuple(
    "TFPMCMCResults",
    [
        "samples",
        "TFPTrace",
        'total_time',
        'time_per_iteration',
        'num_leapfrog_steps',
        'num_leapfrog_steps_per_iter',
    ]
)

def print_variables(variables):
    formatted_values = []
    max_width = 0

    # Format values and find the maximum width
    for name, value in variables.items():
        formatted_value = f"{value:.8g}"
        width = max(len(name), len(formatted_value))
        formatted_values.append((name, formatted_value, width))
        max_width = max(max_width, width)

    # Print the formatted row using the maximum width
    row = "   ".join([f"{name}: {formatted_value:<{max_width}}" for name, formatted_value, width in formatted_values])
    print(row)

# a simple class to make an immutable class pickleable by storing initial arguments
class Immutable:
    def __init__(self,*args,**vargs):
        self.init_args = args
        self.init_vargs = vargs

    def __getstate__(self):
        #print("I'm being pickled")
        return (self.init_args,self.init_vargs)

    def __setstate__(self, d):
        #print("I'm being unpickled with these values: " + repr(d))
        init_args = d[0]
        init_vargs = d[1]
        self.__init__(*init_args,**init_vargs) # just call original init


class timeit:
    def __init__(self,desc='timing',show=True):
        self.desc = desc
        self.show = show
    
    def __enter__(self):
        self.t0 = time.time()
        if self.show:
            print(self.desc+'...',end='',flush=True)
    
    def __exit__(self,*args):
        if self.show:
            print(f"{time.time()-self.t0} sec")


def assert_same(x,y):
    assert jax.tree_util.tree_structure(x) == jax.tree_util.tree_structure(y)
    assert all([x.shape == y.shape for x, y in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y))]), "shapes do not match"
    assert all([x.dtype == y.dtype for x, y in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y))]), "dtypes do not match"
    assert all([jnp.allclose(x,y) for x, y in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y))]), "values do not match"



# compute wasserstein distance for all 1-d marginals
def wass1(A,B):
    assert A.ndim==2
    assert B.ndim==2
    A1 = jnp.sort(A,axis=0)
    B1 = jnp.sort(B,axis=0)
    return jnp.mean(jnp.sum(jnp.abs(A1-B1),axis=1))

rmse = lambda x, y: jnp.sqrt(jnp.mean((x - y)**2))

# @jax.jit
def mean_error(za, zb):
    return rmse(jnp.mean(za, 0), jnp.mean(zb, 0))

# @jax.jit
def cov_error(za, zb):
    return rmse(jnp.cov(za.T), jnp.cov(zb.T))

# @jax.jit
def std_err(za, zb):
    return rmse(jnp.std(za, 0), jnp.std(zb, 0))

# @jax.jit
def normalized_mean_error(za, zb, mean = None, std = None):
    if mean is None:
        mean = jnp.mean(za, 0)
    if std is None:
        std = jnp.std(za, 0)
    assert mean.shape == std.shape
    assert mean.shape[-1] == za.shape[-1]
    return rmse(jnp.mean((za - mean)/std, 0) , jnp.mean((zb - mean)/std, 0))

# @jax.jit
def normalized_cov_error(za, zb, mean = None, std = None):
    if mean is None:
        mean = jnp.mean(za, 0)
    if std is None:
        std = jnp.std(za, 0)
    return rmse(jnp.cov(((za - mean)/std).T), jnp.cov(((zb - mean)/std).T))

# @jax.jit
def normalized_std_err(za, zb, mean = None, std = None):
    if mean is None:
        mean = jnp.mean(za, 0)
    if std is None:
        std = jnp.std(za, 0)
    return rmse(jnp.std((za - mean)/std, 0), jnp.std((zb - mean)/std, 0))

def logsumexp(x, axis):
    return (
        jax.scipy.special.logsumexp(
            x, axis
        )
    )

def stable_log1mexp(a):
    # calculates log (1 - exp (-a))
    # assert a>0
    return jax.lax.cond(
        a <= jnp.log(2),
        lambda a: jnp.log(-jnp.expm1(-a)), 
        lambda a: jnp.log1p(-jnp.exp(-a)), 
        a
    )

def get_estimate_log_var(logR, ddof = 1):
    assert logR.ndim == 1
    N = logR.shape[0]
    assert N > 1
    logRhat = logsumexp(logR, -1) - jnp.log(N)

    def func(logRi):
        return jax.lax.cond(
            logRi>=logRhat,
            lambda logRi: logRi + stable_log1mexp(logRi - logRhat),
            lambda logRi: logRhat + stable_log1mexp(logRhat - logRi), 
            logRi
        )
    return (
        logsumexp(
            jnp.where(
                logR == logRhat, 
                -jnp.inf, 
                2*jax.vmap(func)(logR)
                ),
                -1
            )
        - jnp.log(N - ddof)
    )

def get_log_SNR(logR, ddof = 0):
    assert logR.ndim == 1
    N = logR.shape[0]
    assert N > 1
    logRhat = logsumexp(logR, -1) - jnp.log(N)
    return logRhat - 0.5*get_estimate_log_var(logR, ddof = ddof)



def squareplus(x, gamma = 0.5):
    return 0.5*(x + jnp.sqrt(x**2 + 4*gamma**2))

def squareplus_inv(y, gamma = 1.0):
    return y - gamma/y

def squareplus_prime(x, gamma = 1.0):
    return 0.5*(1 + x/jnp.sqrt(x**2 + 4*gamma**2))

def squareplus_squash(x, a, b, gamma = 1.0):
    return (b-a)*squareplus_prime(x, gamma) + a


def is_scalar(var):
    import jax.numpy as jnp
    import numpy as np
    if isinstance(var, (int, float, bool, complex, jnp.number, np.number)):
        return True
    elif isinstance(var, (np.ndarray, jnp.ndarray)):
        return var.shape == ()
    else:
        return False
