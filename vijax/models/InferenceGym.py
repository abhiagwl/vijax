from inference_gym import using_jax as gym
import jax
import jax.numpy as jnp
import vijax.utils as utils


class InferenceGymWrapper(utils.Immutable):
    """
    A wrapper class for models in the Inference Gym library.
    """

    def __init__(self, model_name, model_args=(), model_kwargs={}):
        """
        Initialize the InferenceGymWrapper with the given model name and arguments.

        Args:
            model_name (str): Name of the model in the Inference Gym library.
            model_args (tuple): Positional arguments for the model.
            model_kwargs (dict): Keyword arguments for the model.
        
        Raises:
            ValueError: If the model is not found in the Inference Gym library.
        """
        if hasattr(gym.targets, model_name):
            self.model = getattr(gym.targets, model_name)(*model_args, **model_kwargs)
            self.vectorized_model = gym.targets.VectorModel(self.model, flatten_sample_transformations=True)
        else:
            raise ValueError(f"Model {model_name} not found in inference_gym")
        self._ndim = self._get_num_latents()
        self._model_name = self.model._pretty_name
        super().__init__(self._ndim)

    @property
    def ndim(self):
        """
        Get the number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        """
        Prevent setting the number of dimensions.

        Args:
            value (int): New number of dimensions.
        
        Raises:
            ValueError: Always, as setting ndim is not allowed.
        """
        raise ValueError("Cannot set ndim")

    def _get_num_latents(self):
        """
        Get the number of latent variables in the model.

        Returns:
            int: Number of latent variables.
        """
        return self.vectorized_model.event_shape[0]

    def log_prob(self, z):
        """
        Compute the log probability of the given latent variable.

        Args:
            z (jnp.ndarray): Latent variable.
        
        Returns:
            jnp.ndarray: Log probability of the latent variable.
        """
        y = self.vectorized_model.default_event_space_bijector(z)
        fldj = self.vectorized_model.default_event_space_bijector.forward_log_det_jacobian(z)
        return self.vectorized_model.unnormalized_log_prob(y) + fldj

    def sample_prior(self, key):
        """
        Sample from the prior distribution.

        Args:
            key (jax.random.PRNGKey): Random key for sampling.
        
        Returns:
            jnp.ndarray: Sampled latent variable.
        """
        return self.model.sample(seed=key)

    def reference_samples(self, nsamps, seed=0, key=None):
        """
        Generate reference samples from the target posterior distribution.

        Args:
            nsamps (int): Number of samples to generate.
            seed (int, optional): Seed for random number generation. Defaults to 0.
            key (jax.random.PRNGKey, optional): Random key for sampling. Defaults to None.
        
        Returns:
            jnp.ndarray: Generated samples.
        
        Raises:
            NotImplementedError: If the model does not have a sample method.
        """
        if hasattr(self.model, 'sample'):
            if key is None:
                key = jax.random.PRNGKey(seed)
            return self.model.sample(sample_shape=(nsamps,), seed=key)
        else:
            raise NotImplementedError(f"Model {self._model_name} does not have a sample method")

    def constrain(self, z):
        """
        Apply constraints to the latent variable.

        Args:
            z (jnp.ndarray): Latent variable.
        
        Returns:
            jnp.ndarray: Constrained latent variable.
        """
        return self.vectorized_model.default_event_space_bijector(z)


class NealsFunnel(InferenceGymWrapper):
    """
    A specific model wrapper for Neal's Funnel.
    """
    def __init__(self, ndim):
        """
        Initialize the Neal's Funnel model with the given number of dimensions.

        Args:
            ndim (int): Number of dimensions.
        """
        super().__init__('NealsFunnel', model_args=(ndim,))


class Banana(InferenceGymWrapper):
    """
    A specific model wrapper for the Banana model.
    """
    def __init__(self, ndim):
        """
        Initialize the Banana model with the given number of dimensions.

        Args:
            ndim (int): Number of dimensions.
        """
        super().__init__('Banana', model_args=(ndim,))


class IllConditionedGaussian(InferenceGymWrapper):
    """
    A specific model wrapper for the Ill-Conditioned Gaussian model.
    """
    def __init__(self, ndim, seed=11):
        """
        Initialize the Ill-Conditioned Gaussian model with the given number of dimensions and seed.

        Args:
            ndim (int): Number of dimensions.
            seed (int): Seed for random number generation.
        """
        super().__init__('IllConditionedGaussian', model_args=(ndim,), model_kwargs={'seed': seed})


class WellConditionedGaussian(InferenceGymWrapper):
    """
    A specific model wrapper for the Well-Conditioned Gaussian model.
    """
    def __init__(self, ndim, seed=11):
        """
        Initialize the Well-Conditioned Gaussian model with the given number of dimensions, seed, and gamma shape parameter.

        Args:
            ndim (int): Number of dimensions.
            seed (int): Seed for random number generation.
        """
        super().__init__('IllConditionedGaussian', model_args=(ndim,), model_kwargs={'seed': seed, 'gamma_shape_parameter': 3.0})
