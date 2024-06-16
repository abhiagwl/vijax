import jax
import jax.numpy as jnp
import vijax.utils as utils
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

class Funana(utils.Immutable):
    def __init__(self, ndim):
        """
        Initialize the Funana model with the given number of dimensions.
        
        Args:
            ndim (int): Number of dimensions. Must be at least 3.
        
        The model is defined as follows:
        - z0 ~ Normal(0, 10)
        - z1 ~ Normal(0, 3)
        - z2, z3, ..., z_{ndim-1} | z0, z1 ~ Normal(curvature * (z0^2 - 100), exp(z1 / 2))
        """
        assert ndim >= 3
        self._ndim = ndim
        self.curvature = 0.03
        self._z0_dist = tfd.Normal(0, 10)
        self._z1_dist = tfd.Normal(0, 3)
        self._z2_dist = lambda z0, z1: tfd.Normal(self.curvature*(z0**2 - 100), jnp.exp(z1/2))
        super().__init__(self.ndim)
    
    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions.
        
        Returns:
            int: Number of dimensions.
        """
        return self._ndim
    
    @ndim.setter
    def ndim(self, ndim: int):
        """
        Set the number of dimensions. This setter does not actually change the value.
        
        Args:
            ndim (int): Number of dimensions.
        
        Returns:
            int: The current number of dimensions.
        """
        return self._ndim
    
    def log_prob(self, z):
        """
        Compute the log probability of the given latent variable.
        
        Args:
            z (jnp.ndarray): Latent variable of shape (ndim,).
        
        Returns:
            jnp.ndarray: Log probability of the latent variable.
        
        The log probability is computed as:
        log_prob(z) = log_prob(z0) + log_prob(z1) + sum(log_prob(z2, z3, ..., z_{ndim-1} | z0, z1))
        where:
        - z0 ~ Normal(0, 10)
        - z1 ~ Normal(0, 3)
        - z2, z3, ..., z_{ndim-1} | z0, z1 ~ Normal(curvature * (z0^2 - 100), exp(z1 / 2))
        
        Note: The log_prob method accepts samples in the unconstrained space.
        """
        assert z.shape == (self.ndim,)
        x_0 = z[0]
        x_1 = z[1]
        x_2 = z[2:]
        return (
            self._z0_dist.log_prob(x_0) 
            + self._z1_dist.log_prob(x_1)
            + self._z2_dist(x_0, x_1).log_prob(x_2).sum() 
        )

    def sample_prior(self, PRNGKey = jax.random.PRNGKey(0)):
        """
        Sample from the prior distribution.
        
        Args:
            PRNGKey (jax.random.PRNGKey): Random key for sampling.
        
        Returns:
            jnp.ndarray: Sampled latent variable of shape (ndim,).
        
        The sampling process is as follows:
        - Sample z0 from Normal(0, 10)
        - Sample z1 from Normal(0, 3)
        - Sample z2, z3, ..., z_{ndim-1} from Normal(curvature * (z0^2 - 100), exp(z1 / 2))
        
        Note: The sample_prior method outputs samples in the unconstrained space.
        """
        key, subkey = jax.random.split(PRNGKey)
        x_0 = self._z0_dist.sample(seed=subkey)
        key, subkey = jax.random.split(key)
        x_1 = self._z1_dist.sample(seed=subkey)
        key, subkey = jax.random.split(key)
        x_2 = self._z2_dist(x_0, x_1).sample(seed=subkey, sample_shape=(self.ndim - 2,))
        _z = jnp.concatenate([jnp.array([x_0]), jnp.array([x_1]), x_2])
        assert _z.shape == (self.ndim,)
        return _z
    
    def reference_samples(self, nsamps, seed = 0, key = None):
        """
        Generate reference samples from the target posterior distribution.
        
        Args:
            nsamps (int): Number of samples to generate.
            seed (int, optional): Seed for random number generation. Defaults to 0.
            key (jax.random.PRNGKey, optional): Random key for sampling. Defaults to None.
        
        Returns:
            jnp.ndarray: Generated samples of shape (nsamps, ndim).
        
        Note: The reference_samples method outputs samples in the unconstrained space.
        """
        if seed is None:
            assert key is not None
        if key is None:
            assert seed is not None
            key = jax.random.PRNGKey(seed)
        _sampler = lambda key: self.sample_prior(key)
        fast_sampler = jax.jit(jax.vmap(_sampler))
        return fast_sampler(jax.random.split(key, nsamps))    
    
    def constrain(self, z):
        """
        Apply constraints to the latent variable. This method currently does nothing.
        
        Args:
            z (jnp.ndarray): Latent variable.
        
        Returns:
            jnp.ndarray: Constrained latent variable (unchanged).
        """
        return z
