# @Gaussian.py: This module defines a Gaussian variational distribution with various utility functions for parameter transformations and sampling.
import jax
import jax.numpy as jnp
import jax.scipy as jscipy

def log_add_exp(x1, x2):
    """
    Compute the log of the sum of exponentials of inputs.

    Args:
        x1 (jnp.ndarray): First input array.
        x2 (jnp.ndarray): Second input array.

    Returns:
        jnp.ndarray: Result of log(exp(x1) + exp(x2)).
    """
    return jnp.maximum(x1, x2) + jnp.log1p(jnp.exp(-jnp.abs(x1-x2)))

def log_sub_exp(x1, x2):
    """
    Compute the log of the difference of exponentials of inputs.

    Args:
        x1 (jnp.ndarray): First input array.
        x2 (jnp.ndarray): Second input array.

    Returns:
        jnp.ndarray: Result of log(exp(x1) - exp(x2)).
    """
    return x1 + jnp.log1p(-jnp.exp(x2-x1))

def proximal_forward(x, gamma=1):
    """
    Apply the proximal forward transformation.

    Args:
        x (jnp.ndarray): Input array.
        gamma (float, optional): Scaling parameter. Defaults to 1.

    Returns:
        jnp.ndarray: Transformed array.
    """
    return 0.5 * (x + jnp.sqrt(x**2 + 4*gamma))

def proximal_backward(x, gamma=1):
    """
    Apply the proximal backward transformation.

    Args:
        x (jnp.ndarray): Input array.
        gamma (float, optional): Scaling parameter. Defaults to 1.

    Returns:
        jnp.ndarray: Transformed array.
    """
    return (x**2 - gamma) / x

def pos_diag(x):
    """
    Ensure the diagonal elements are positive.

    Args:
        x (jnp.ndarray): Input array.

    Returns:
        jnp.ndarray: Transformed array with positive diagonal elements.
    """
    assert x.ndim == 1
    return log_add_exp(x, 0)

def pos_tril(x):
    """
    Ensure the lower triangular part of the matrix is positive.

    Args:
        x (jnp.ndarray): Input matrix.

    Returns:
        jnp.ndarray: Transformed matrix with positive lower triangular part.
    """
    assert x.ndim == 2
    return jnp.tril(x, -1) + jnp.diag(pos_diag(jnp.diag(x)))

def inv_pos_diag(x):
    """
    Inverse transformation to ensure the diagonal elements are positive.

    Args:
        x (jnp.ndarray): Input array.

    Returns:
        jnp.ndarray: Inverse transformed array with positive diagonal elements.
    """
    assert x.ndim == 1
    return log_sub_exp(x, 0)

def inv_pos_tril(x):
    """
    Inverse transformation to ensure the lower triangular part of the matrix is positive.

    Args:
        x (jnp.ndarray): Input matrix.

    Returns:
        jnp.ndarray: Inverse transformed matrix with positive lower triangular part.
    """
    assert x.ndim == 2
    return jnp.tril(x, -1) + jnp.diag(inv_pos_diag(jnp.diag(x)))

def flip(A):
    """
    Flip the matrix upside down and left to right.

    Args:
        A (jnp.ndarray): Input matrix.

    Returns:
        jnp.ndarray: Flipped matrix.
    """
    return jnp.flipud(jnp.fliplr(A))

def hessian_to_factor(H):
    """
    Given a Hessian of a target distribution, find a factor for a covariance matrix.

    Args:
        H (jnp.ndarray): Hessian matrix.

    Returns:
        jnp.ndarray: Factor for the covariance matrix.
    """
    L = jnp.linalg.cholesky(flip(H))
    Li = jscipy.linalg.solve_triangular(L, jnp.eye(H.shape[0]), lower=True)
    return flip(Li.T)

class Gaussian:
    """
    Gaussian variational distribution class.
    """

    S = jnp.log(jnp.exp(1) - 1)  # Used to initialize the covariance matrix to identity

    def __init__(self, ndim):
        """
        Initialize the Gaussian distribution with the given number of dimensions.

        Args:
            ndim (int): Number of dimensions.
        """
        self._ndim = ndim

    @property
    def ndim(self):
        """
        Get the number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        return self._ndim

    @ndim.setter
    def ndim(self, ndim):
        """
        Set the number of dimensions. Ensures the value is non-negative.

        Args:
            ndim (int): Number of dimensions.

        Raises:
            ValueError: If ndim is negative.
        """
        if ndim < 0:
            raise ValueError("ndim must be non-negative")
        self._ndim = ndim

    def initial_params(self):
        """
        Initialize the parameters of the Gaussian distribution.

        Returns:
            VarParams: Initial parameters.
        """
        return [jnp.zeros(self.ndim).astype(float),
                self.S * jnp.eye(self.ndim).astype(float)]

    def transform_params(self, params):
        """
        Transform the parameters to ensure they are valid.

        Args:
            params (VarParams): Input parameters.

        Returns:
            VarParams: Transformed parameters.
        """
        return params[0], pos_tril(params[1])

    def sample(self, params, key = jax.random.PRNGKey(0)):
        """
        Sample from the Gaussian distribution.

        Args:
            params (VarParams): Parameters of the distribution.
            key (Key, optional): Random key for sampling. Defaults to jax.random.PRNGKey(0).

        Returns:
            Latent: Sampled latent variable.
        """
        mu, sig = self.transform_params(params)
        ε = jax.random.normal(key, (self.ndim,))
        return mu + jnp.dot(ε, sig.T)

    def log_prob(self, params, sample):
        """
        Compute the log probability of a sample.

        Args:
            params (VarParams): Parameters of the distribution.
            sample (Latent): Sampled latent variable.

        Returns:
            Scalar: Log probability of the sample.
        """
        mu, sig = self.transform_params(params)
        Λ = jscipy.linalg.solve_triangular(sig, jnp.eye(self.ndim), lower=True).T
        M = (sample - mu)
        a = self.ndim * jnp.log(2 * jnp.pi)
        b = 2 * jnp.sum(jnp.log((jnp.diag(sig))))
        c = jnp.sum(jnp.matmul(M, jnp.matmul(Λ, Λ.T)) * M, -1)
        return -0.5 * (a + b + c)

    def entropy(self, params):
        """
        Compute the entropy of the Gaussian distribution.

        Args:
            params (VarParams): Parameters of the distribution.

        Returns:
            Scalar: Entropy of the distribution.
        """
        _, sig = self.transform_params(params)
        a = self.ndim * jnp.log(2 * jnp.pi) + self.ndim
        b = 2 * jnp.sum(jnp.log((jnp.diag(sig))))
        return 0.5 * (a + b)

    def match_mean_inv_cov(self, mean, inv_cov):
        """
        Match the mean and inverse covariance of the distribution.

        Args:
            mean (jnp.ndarray): Mean of the distribution.
            inv_cov (jnp.ndarray): Inverse covariance matrix.

        Returns:
            Tuple[Gaussian, VarParams]: Updated Gaussian distribution and parameters.
        """
        C = hessian_to_factor(inv_cov)
        return self, (mean, inv_pos_tril(C))

    def match_mean_cov(self, mean, cov):
        """
        Match the mean and covariance of the distribution.

        Args:
            mean (jnp.ndarray): Mean of the distribution.
            cov (jnp.ndarray): Covariance matrix.

        Returns:
            Tuple[Gaussian, VarParams]: Updated Gaussian distribution and parameters.
        """
        C = jnp.linalg.cholesky(cov)
        return self, (mean, inv_pos_tril(C))

    def _log_prob_from_eps(self, params, eps):
        """
        Compute the log probability from epsilon.

        Args:
            params (VarParams): Parameters of the distribution.
            eps (jnp.ndarray): Epsilon values.

        Returns:
            Scalar: Log probability.
        """
        mu, sig = self.transform_params(params)
        a = self.ndim * jnp.log(2 * jnp.pi)
        b = 2 * jnp.sum(jnp.log((jnp.diag(sig))))
        c = jnp.sum(eps**2, -1)
        return -0.5 * (a + b + c)

    def sample_and_log_prob(self, params, key = jax.random.PRNGKey(0)):
        """
        Sample from the distribution and compute the log probability.

        Args:
            params (VarParams): Parameters of the distribution.
            key (Key, optional): Random key for sampling. Defaults to jax.random.PRNGKey(0).

        Returns:
            Tuple[Latent, Scalar]: Sampled latent variable and log probability.
        """
        mu, sig = self.transform_params(params)
        ε = jax.random.normal(key, (self.ndim,))
        z = mu + jnp.dot(ε, sig.T)
        return z, self._log_prob_from_eps(params, ε)

    def sample_and_log_prob_stl(self, params, key = jax.random.PRNGKey(0)):
        """
        Sample from the distribution and compute the log probability using stop-gradient.

        Args:
            params (VarParams): Parameters of the distribution.
            key (Key, optional): Random key for sampling. Defaults to jax.random.PRNGKey(0).

        Returns:
            Tuple[Latent, Scalar]: Sampled latent variable and log probability.
        """
        mu, sig = self.transform_params(params)
        ε = jax.random.normal(key, (self.ndim,))
        z = mu + jnp.dot(ε, sig.T)
        _sig = jax.lax.stop_gradient(sig)
        _mu = jax.lax.stop_gradient(mu)
        _ε = jax.scipy.linalg.solve_triangular(_sig, z - _mu, lower=True)
        return z, self._log_prob_from_eps(jax.lax.stop_gradient(params), _ε)
