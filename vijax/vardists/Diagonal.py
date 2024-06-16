from vijax.vardists.Gaussian import Gaussian
import jax.numpy as jnp
from vijax.vardists.Gaussian import inv_pos_tril, pos_diag

class Diagonal(Gaussian):
    """
    A class to implement Diagonal Gaussian distribution.
    Inherits from the Gaussian class.
    """

    def initial_params(self):
        """
        Initialize the parameters of the Diagonal Gaussian distribution.

        Returns:
            List[jnp.ndarray]: A list containing the initial mean (zeros) and 
                               initial diagonal covariance (scaled identity matrix).
        """
        return [jnp.zeros(self.ndim,).astype(float),
                self.S * jnp.ones(self.ndim,).astype(float)]

    def transform_params(self, params):
        """
        Transform the parameters to the required format.

        Args:
            params (List[jnp.ndarray]): List containing the mean and diagonal covariance.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Transformed mean and diagonal covariance matrix.
        """
        return params[0], jnp.diag(pos_diag(params[1]))

    def match_mean_cov(self, mean, cov):
        """
        Match the given mean and covariance to the Diagonal Gaussian distribution.

        Args:
            mean (jnp.ndarray): The mean vector.
            cov (jnp.ndarray): The covariance matrix.

        Returns:
            Tuple[Diagonal, Tuple[jnp.ndarray, jnp.ndarray]]: The Diagonal distribution instance 
                                                              and the matched mean and diagonal covariance.
        """
        C = jnp.linalg.cholesky(cov)
        return self, (mean, jnp.diag(inv_pos_tril(C)))
