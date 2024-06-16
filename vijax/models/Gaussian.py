import jax
import jax.numpy as jnp

from vijax.utils import Immutable

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

class QuickGaussian(Immutable):
    def __init__(self, loc, scale_tril, ndim):
        """
        Initialize the QuickGaussian model with the given location, scale, and number of dimensions.
        
        Args:
            loc (jnp.ndarray): Mean of the Gaussian distribution.
            scale_tril (jnp.ndarray): Lower triangular matrix for the scale of the Gaussian distribution.
            ndim (int): Number of dimensions.
        """
        self.dist = tfd.MultivariateNormalTriL(
            loc=loc,
            scale_tril=scale_tril
        )
        self._ndim = ndim
        super().__init__(ndim)  # calls Immutable.__init__(self, ndim)
    
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

    def log_prob(self, z):
        """
        Compute the log probability of the given latent variable.
        
        Args:
            z (jnp.ndarray): Latent variable of shape (ndim,).
        
        Returns:
            jnp.ndarray: Log probability of the latent variable.
        """
        assert z.shape == (self.ndim,)
        return jnp.sum(self.dist.log_prob(z))
    
    def sample_prior(self, PRNGKey=jax.random.PRNGKey(0)):
        """
        Sample from the prior distribution.
        
        Args:
            PRNGKey (jax.random.PRNGKey): Random key for sampling.
        
        Returns:
            jnp.ndarray: Sampled latent variable of shape (ndim,).
        """
        return self.dist.sample(seed=PRNGKey)

    def reference_samples(self, nsamps, key=None):
        """
        Generate reference samples from the target posterior distribution.
        
        Args:
            nsamps (int): Number of samples to generate.
            key (jax.random.PRNGKey, optional): Random key for sampling. Defaults to None.
        
        Returns:
            jnp.ndarray: Generated samples of shape (nsamps, ndim).
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        return self.dist.sample(seed=key, sample_shape=(nsamps,))

    def constrain(self, z):
        """
        Apply constraints to the latent variable. This method currently does nothing.
        
        Args:
            z (jnp.ndarray): Latent variable.
        
        Returns:
            jnp.ndarray: Constrained latent variable (unchanged).
        """
        return z