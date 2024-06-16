import jax
import jax.numpy as jnp

import numpyro.distributions as num_dists

class Studentt:
    def __init__(self, ndim, df = 1.5):
        """
        Initialize the Studentt model with the given number of dimensions and degrees of freedom.
        
        Args:
            ndim (int): Number of dimensions.
            df (float): Degrees of freedom for the Student-t distribution.
        """
        self._ndim = ndim
        self.df = df
        self.dist = num_dists.MultivariateStudentT(
                df = self.df, 
                loc = jnp.zeros(self.ndim), 
                scale_tril = jnp.eye(self.ndim)
                )
    
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
        """
        assert z.shape==(self.ndim,)
        return self.dist.log_prob(z)

    def sample_prior(self, PRNGKey = jax.random.PRNGKey(0)):
        """
        Sample from the prior distribution.
        
        Args:
            PRNGKey (jax.random.PRNGKey): Random key for sampling.
        
        Returns:
            jnp.ndarray: Sampled latent variable of shape (ndim,).
        """
        key, subkey = jax.random.split(PRNGKey)
        _z = self.dist.sample(subkey)
        assert _z.shape==(self.ndim,)
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


class Studentt_1_5(Studentt):
    def __init__(self, ndim):
        """
        Initialize the Studentt_1_5 model with the given number of dimensions.
        
        Args:
            ndim (int): Number of dimensions.
        """
        super().__init__(ndim, df = 1.5)

class Studentt_2_5(Studentt):
    def __init__(self, ndim):
        """
        Initialize the Studentt_2_5 model with the given number of dimensions.
        
        Args:
            ndim (int): Number of dimensions.
        """
        super().__init__(ndim, df = 2.5)