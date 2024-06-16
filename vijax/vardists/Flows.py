import jax.numpy as jnp
import jax

from vijax.vardists.Gaussian import Gaussian

import vijax.utils as utils

class Flows():
    """
    Base class for flow-based models.
    """

    def __init__(self, ndim, base_dist=None, base_dist_params=None):
        """
        Initialize the flow model.

        Parameters:
        ndim (int): Dimensionality of the data.
        base_dist (object, optional): Base distribution object. Defaults to Gaussian.
        base_dist_params (dict, optional): Parameters for the base distribution. Defaults to None.
        """
        self._ndim = ndim

        if base_dist is None:
            self.base_dist = Gaussian(self.ndim)
        else:
            self.base_dist = base_dist
        if base_dist_params is None:
            self.base_dist_params = self.base_dist.initial_params()
        else:
            self.base_dist_params = base_dist_params

    @property
    def ndim(self) -> int:
        """
        Get the dimensionality of the data.

        Returns:
        int: Dimensionality of the data.
        """
        return self._ndim

    @ndim.setter
    def ndim(self, ndim: int):
        """
        Set the dimensionality of the data.

        Parameters:
        ndim (int): Dimensionality of the data.

        Raises:
        ValueError: If ndim is negative.
        """
        if ndim < 0:
            raise ValueError("ndim must be non-negative")
        self._ndim = ndim

    def forward_transform(self, params, z):
        """
        Forward transformation function to be implemented by subclasses.

        Parameters:
        params (dict): Parameters for the transformation.
        z (array): Input data.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def inverse_transform(self, params, x):
        """
        Inverse transformation function to be implemented by subclasses.

        Parameters:
        params (dict): Parameters for the transformation.
        x (array): Input data.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def sample(self, params, key):
        """
        Sample from the flow model.

        Parameters:
        params (dict): Parameters for the transformation.
        key (jax.random.PRNGKey): Random key for sampling.

        Returns:
        array: Samples from the flow model.
        """
        z_o = self.base_dist.sample(self.base_dist_params, key)
        samples, neg_log_det_J = self.forward_transform(params, z_o)
        return samples

    def log_prob(self, params, z):
        """
        Compute the log probability of the data under the flow model.

        Parameters:
        params (dict): Parameters for the transformation.
        z (array): Input data.

        Returns:
        array: Log probability of the data.
        """
        z_o, neg_log_det_J = self.inverse_transform(params, z)
        lq = self.base_dist.log_prob(self.base_dist_params, z_o)
        return lq + neg_log_det_J

    def sample_and_log_prob(self, params, key):
        """
        Sample from the flow model and compute the log probability.

        Parameters:
        params (dict): Parameters for the transformation.
        key (jax.random.PRNGKey): Random key for sampling.

        Returns:
        tuple: Samples and log probability.
        """
        z_o, lq = self.base_dist.sample_and_log_prob(self.base_dist_params, key)
        samples, neg_log_det_J = self.forward_transform(params, z_o)
        return samples, lq + neg_log_det_J

    def sample_and_log_prob_stl(self, params, key):
        """
        Sample from the flow model and compute the log probability using the STL method.

        Parameters:
        params (dict): Parameters for the transformation.
        key (jax.random.PRNGKey): Random key for sampling.

        Returns:
        tuple: Samples and log probability.
        """
        ε = self.base_dist.sample(self.base_dist_params, key)
        z, _ = self.forward_transform(params, ε)

        ε_prime, neg_log_det_J = self.inverse_transform(jax.lax.stop_gradient(params), z)
        lq_0 = self.base_dist.log_prob(self.base_dist_params, ε_prime)

        return z, lq_0 + neg_log_det_J
    
def BinaryFlip(z, j):
    """
    Flip the binary split of the input data.

    Parameters:
    z (array): Input data.
    j (int): Index for splitting.

    Returns:
    array: Flipped data.
    """
    splitted = BinarySplit(z, j)
    return jnp.concatenate([splitted[1],splitted[0]], -1)

def ReverseBinaryFlip(z, j):
    """
    Reverse the binary flip of the input data.

    Parameters:
    z (array): Input data.
    j (int): Index for splitting.

    Returns:
    array: Reversed flipped data.
    """
    splitted = ReverseBinarySplit(z, j)
    return jnp.concatenate([splitted[1],splitted[0]], -1)

def BinarySplit(z, j):
    """
    Split the input data into two parts.

    Parameters:
    z (array): Input data.
    j (int): Index for splitting.

    Returns:
    tuple: Split data.
    """
    D = z.shape[-1]
    d = D//2
    if D % 2 == 1:
        d += (int(j) % 2)
    return jnp.array_split(z, [d], -1)

def ReverseBinarySplit(z, j):
    """
    Reverse the binary split of the input data.

    Parameters:
    z (array): Input data.
    j (int): Index for splitting.

    Returns:
    tuple: Reversed split data.
    """
    return BinarySplit(z, int(j)+1)

##############################################################################
# Generic Flow Utilities
##############################################################################

def coupling_layer_specifications(num_hidden_units, num_hidden_layers, z_len):
    """
    Specify the coupling layer sizes for the flow model.

    Parameters:
    num_hidden_units (int): Number of hidden units in each layer.
    num_hidden_layers (int): Number of hidden layers.
    z_len (int): Length of the input data.

    Returns:
    list: Coupling layer sizes.
    """
    d_1 = int(z_len//2)
    d_2 = int(z_len - d_1)
    coupling_layer_sizes = []
    coupling_layer_sizes.append([d_1] +
                                num_hidden_layers*[num_hidden_units] + [2*d_2])
    coupling_layer_sizes.append([d_2] +
                                num_hidden_layers*[num_hidden_units] + [2*d_1])
    return coupling_layer_sizes

##############################################################################
# Generic NN Utilities
##############################################################################

def relu(x):
    """
    ReLU activation function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Activated data.
    """
    return jnp.maximum(0, x)

def leakyrelu(x, slope=0.01):
    """
    Leaky ReLU activation function.

    Parameters:
    x (array): Input data.
    slope (float, optional): Slope for the negative part. Defaults to 0.01.

    Returns:
    array: Activated data.
    """
    return jnp.maximum(0, x) + slope*jnp.minimum(0, x)

def tanh(x):
    """
    Tanh activation function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Activated data.
    """
    return jnp.tanh(x)

def softmax_matrix(x):
    """
    Softmax function applied to a matrix.

    Parameters:
    x (array): Input data.

    Returns:
    array: Softmax activated data.
    """
    z = x - jnp.max(x, axis=-1, keepdims=True)
    numerator = jnp.exp(z)
    denominator = jnp.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator

def offset_squareplus(gamma, pad):
    """
    Offset for the squareplus function.

    Parameters:
    gamma (float): Gamma parameter.
    pad (float): Padding value.

    Returns:
    float: Offset value.
    """
    p = 1 - pad
    return (gamma**2 - p**2)/(p)

def offset_softplus(pad):
    """
    Offset for the softplus function.

    Parameters:
    pad (float): Padding value.

    Returns:
    float: Offset value.
    """
    p = 1 - pad
    return -(p + utils.stable_log1mexp(p))

def offset_exp(pad):
    """
    Offset for the exponential function.

    Parameters:
    pad (float): Padding value.

    Returns:
    float: Offset value.
    """
    p = 1 - pad
    return -(jnp.log(p))

##############################################################################
# RealNVP
##############################################################################

def exp_scale_function(x):
    """
    Exponential scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Scaled data.
    """
    return jnp.exp(x)

def log_exp_scale_function(x):
    """
    Logarithm of the exponential scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Log-scaled data.
    """
    return x

def padded_exp_scale_function(x, pad = 0.1):
    """
    Padded exponential scale function.

    Parameters:
    x (array): Input data.
    pad (float, optional): Padding value. Defaults to 0.1.

    Returns:
    array: Padded scaled data.
    """
    offset = offset_exp(pad)
    return jnp.exp(x - offset) + pad

def log_padded_exp_scale_function(x, pad = 0.1):
    """
    Logarithm of the padded exponential scale function.

    Parameters:
    x (array): Input data.
    pad (float, optional): Padding value. Defaults to 0.1.

    Returns:
    array: Log-padded scaled data.
    """
    offset = offset_exp(pad)
    return jnp.logaddexp(x - offset, jnp.log(pad))

def softplus_scale_function(x):
    """
    Softplus scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Scaled data.
    """
    return jax.nn.softplus(x)

def log_softplus_scale_function(x):
    """
    Logarithm of the softplus scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Log-scaled data.
    """
    return jnp.log(jax.nn.softplus(x))

def padded_softplus_scale_function(x, pad = 0.1):
    """
    Padded softplus scale function.

    Parameters:
    x (array): Input data.
    pad (float, optional): Padding value. Defaults to 0.1.

    Returns:
    array: Padded scaled data.
    """
    offset = offset_softplus(pad)
    return jax.nn.softplus(x - offset) + pad

def log_padded_softplus_scale_function(x, pad = 0.1):
    """
    Logarithm of the padded softplus scale function.

    Parameters:
    x (array): Input data.
    pad (float, optional): Padding value. Defaults to 0.1.

    Returns:
    array: Log-padded scaled data.
    """
    offset = offset_softplus(pad)
    return jnp.log(jax.nn.softplus(x - offset) + pad)

def squareplus_scale_function(x):
    """
    Squareplus scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Scaled data.
    """
    gamma = 1.0
    return utils.squareplus(x, gamma)

def log_squareplus_scale_function(x):
    """
    Logarithm of the squareplus scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Log-scaled data.
    """
    gamma = 1.0
    return jnp.log(utils.squareplus(x, gamma))

def padded_squareplus_scale_function(x, pad = 0.1):
    """
    Padded squareplus scale function.

    Parameters:
    x (array): Input data.
    pad (float, optional): Padding value. Defaults to 0.1.

    Returns:
    array: Padded scaled data.
    """
    gamma = 1.0
    offset = offset_squareplus(gamma, pad)
    return utils.squareplus(x - offset, gamma) + pad

def log_padded_squareplus_scale_function(x, pad = 0.1):
    """
    Logarithm of the padded squareplus scale function.

    Parameters:
    x (array): Input data.
    pad (float, optional): Padding value. Defaults to 0.1.

    Returns:
    array: Log-padded scaled data.
    """
    gamma = 1.0
    offset = offset_squareplus(gamma, pad)
    return jnp.log(utils.squareplus(x - offset, gamma) + pad)

def tanh_scale_function(x):
    """
    Tanh scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Scaled data.
    """
    return jnp.exp(jnp.tanh(x))

def log_tanh_scale_function(x):
    """
    Logarithm of the tanh scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Log-scaled data.
    """
    return jnp.tanh(x)

def scaled_tanh_scale_function(x, scale = 0.5):
    """
    Scaled tanh scale function.

    Parameters:
    x (array): Input data.
    scale (float, optional): Scaling factor. Defaults to 0.5.

    Returns:
    array: Scaled data.
    """
    return jnp.exp(scale*jnp.tanh(x))

def log_scaled_tanh_scale_function(x, scale = 0.5):
    """
    Logarithm of the scaled tanh scale function.

    Parameters:
    x (array): Input data.
    scale (float, optional): Scaling factor. Defaults to 0.5.

    Returns:
    array: Log-scaled data.
    """
    return scale*jnp.tanh(x)

def larged_tanh_scale_function(x, scale = 2.0):
    """
    Larger scaled tanh scale function.

    Parameters:
    x (array): Input data.
    scale (float, optional): Scaling factor. Defaults to 2.0.

    Returns:
    array: Scaled data.
    """
    return jnp.exp(scale*jnp.tanh(x))

def log_larged_tanh_scale_function(x, scale = 2.0):
    """
    Logarithm of the larger scaled tanh scale function.

    Parameters:
    x (array): Input data.
    scale (float, optional): Scaling factor. Defaults to 2.0.

    Returns:
    array: Log-scaled data.
    """
    return scale*jnp.tanh(x)

def squaresquash_scale_function(x):
    """
    Squareplus squash scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Scaled data.
    """
    return jnp.exp(utils.squareplus_squash(x, -1, 1, 1.0))

def log_squaresquash_scale_function(x):
    """
    Logarithm of the squareplus squash scale function.

    Parameters:
    x (array): Input data.

    Returns:
    array: Log-scaled data.
    """
    return utils.squareplus_squash(x, -1, 1, 1.0)

def scaled_squaresquash_scale_function(x, scale = 0.5):
    """
    Scaled squareplus squash scale function.

    Parameters:
    x (array): Input data.
    scale (float, optional): Scaling factor. Defaults to 0.5.

    Returns:
    array: Scaled data.
    """
    return jnp.exp(scale*utils.squareplus_squash(x, -1, 1, 1.0))

def log_scaled_squaresquash_scale_function(x, scale = 0.5):
    """
    Logarithm of the scaled squareplus squash scale function.

    Parameters:
    x (array): Input data.
    scale (float, optional): Scaling factor. Defaults to 0.5.

    Returns:
    array: Log-scaled data.
    """
    return scale*utils.squareplus_squash(x, -1, 1, 1.0)

def get_scale_function(scale_func):
    """
    Get the scale function and its logarithm based on the function name.

    Parameters:
    scale_func (str): Name of the scale function.

    Returns:
    tuple: Scale function and its logarithm.

    Raises:
    NotImplementedError: If the scale function name is not recognized.
    """
    if scale_func == "exp":
        return exp_scale_function, log_exp_scale_function
    elif scale_func == "padded_exp":
        return padded_exp_scale_function, log_padded_exp_scale_function
    elif scale_func == "softplus":
        return softplus_scale_function, log_softplus_scale_function
    elif scale_func == "padded_softplus":
        return padded_softplus_scale_function, log_padded_softplus_scale_function
    elif scale_func == "squareplus":
        return squareplus_scale_function, log_squareplus_scale_function
    elif scale_func == "padded_squareplus":
        return padded_squareplus_scale_function, log_padded_squareplus_scale_function
    elif scale_func == "tanh":
        return tanh_scale_function, log_tanh_scale_function
    elif scale_func == "scaled_tanh":
        return scaled_tanh_scale_function, log_scaled_tanh_scale_function
    elif scale_func == "larged_tanh":
        return larged_tanh_scale_function, log_larged_tanh_scale_function
    elif scale_func == "squaresquash":
        return squaresquash_scale_function, log_squaresquash_scale_function
    elif scale_func == "scaled_squaresquash":
        return scaled_squaresquash_scale_function, log_scaled_squaresquash_scale_function
    else:
        print(f"Expected one of [exp, padded_exp, softplus, padded_softplus, squareplus, padded_squareplus, tanh, scaled_tanh, larged_tanh, squaresquash, scaled_squaresquash], got {scale_func}")
        raise NotImplementedError

class RealNVP(Flows):
    """
    RealNVP flow model.
    """

    def __init__(
                self,
                ndim,
                num_transformations,
                num_hidden_units,
                num_hidden_layers,
                params_init_scale,
                scale_func_name = "tanh",
                base_dist=None,
                base_dist_params=None
                ):
        """
        Initialize the RealNVP model.

        Parameters:
        ndim (int): Dimensionality of the data.
        num_transformations (int): Number of transformations.
        num_hidden_units (int): Number of hidden units in each layer.
        num_hidden_layers (int): Number of hidden layers.
        params_init_scale (float): Initial scale for the parameters.
        scale_func_name (str): Name of the scale function. Defaults to "tanh".
        base_dist (Distribution, optional): Base distribution. Defaults to None.
        base_dist_params (dict, optional): Parameters for the base distribution. Defaults to None.
        """
        self.num_transformations = num_transformations
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.params_init_scale = params_init_scale
        self.scale_func_name = scale_func_name
        self.scale_function, self.log_scale_function = get_scale_function(self.scale_func_name)
        
        super(RealNVP, self).__init__(ndim=ndim, base_dist=base_dist, base_dist_params=base_dist_params)

    def mean_cov(self, params):
        """
        Compute the mean and covariance of the distribution.

        Parameters:
        params (list): Parameters of the model.

        Raises:
        NotImplementedError: This method should be implemented in a subclass.
        """
        raise NotImplementedError

    def match_mean_inv_cov(self, mean, inv_cov):
        """
        Match a given mean and inverse covariance by adjusting the base distribution.

        Parameters:
        mean (array): Desired mean.
        inv_cov (array): Desired inverse covariance matrix.

        Returns:
        tuple: New distribution and its initial parameters.
        """
        new_base_dist, new_base_dist_params = self.base_dist.match_mean_inv_cov(mean, inv_cov)
        new_dist = RealNVP(
            self.ndim, 
            self.num_transformations,
            self.num_hidden_units,
            self.num_hidden_layers,
            self.params_init_scale,
            self.scale_func_name,
            new_base_dist,
            new_base_dist_params
        )
        new_params = new_dist.initial_params()
        return new_dist, new_params

    def match_mean_cov(self, mean, cov):
        """
        Match a given mean and covariance by adjusting the base distribution.

        Parameters:
        mean (array): Desired mean.
        cov (array): Desired covariance matrix.

        Returns:
        tuple: New distribution and its initial parameters.
        """
        new_base_dist, new_base_dist_params = self.base_dist.match_mean_cov(mean, cov)
        new_dist = RealNVP(
            self.ndim, 
            self.num_transformations,
            self.num_hidden_units,
            self.num_hidden_layers,
            self.params_init_scale,
            self.scale_func_name,
            new_base_dist,
            new_base_dist_params
        )
        new_params = new_dist.initial_params()
        return new_dist, new_params

    def initial_params(self):
        """
        Generate the initial parameters for the model.

        Returns:
        list: Initial parameters for the model.
        """
        def generate_net_st():
            """
            Generate the parameters for a single coupling layer.

            Returns:
            list: Parameters for the coupling layer.
            """
            key = jax.random.PRNGKey(0)

            coupling_layer_sizes = coupling_layer_specifications(
                                self.num_hidden_units,
                                self.num_hidden_layers, self.ndim)

            init_st_params = []

            for layer_sizes in coupling_layer_sizes:
                key1, key2, key = jax.random.split(key, 3)
                init_st_params.append([(
                    self.params_init_scale * jax.random.normal(key1, (m, n)),   # weight matrix
                    self.params_init_scale * jax.random.normal(key2, (n,)))      # bias vector
                    for m, n in
                    zip(layer_sizes[:-1], layer_sizes[1:])])
            return init_st_params

        st = [generate_net_st() for i in range(self.num_transformations)]

        return st

    def apply_net_st(self, params, inputs):
        """
        Apply the coupling layer to the inputs.

        Parameters:
        params (list): Parameters of the coupling layer.
        inputs (array): Input data.

        Returns:
        tuple: Scale and translation components.
        """
        inpW, inpb = params[0]

        inputs = leakyrelu(jnp.dot(inputs, inpW) + inpb)

        for W, b in params[1:-1]:
            outputs = jnp.dot(inputs, W) + b

        outW, outb = params[-1]
        outputs = jnp.dot(inputs, outW) + outb

        assert(outputs.shape[:-1] == inputs.shape[:-1])
        assert(outputs.shape[-1] % 2 == 0)

        s, t = jnp.array_split(outputs, 2, -1)

        assert(s.shape == t.shape)

        return s, t

    def forward_transform(self, params, z):
        """
        Apply the forward transformation to the data.

        Parameters:
        params (list): Parameters of the model.
        z (array): Input data.

        Returns:
        tuple: Transformed data and the negative log determinant of the Jacobian.
        """
        neg_log_det_J = jnp.zeros(z.shape[:-1])

        st_list = params

        for i in range(self.num_transformations):
            for j in range(2):

                z_1, z_2 = BinarySplit(z, j)

                s, t = self.apply_net_st(params=st_list[i][j], inputs=z_1)
                z = jnp.concatenate([z_1, z_2*self.scale_function(s) + t], axis=-1)
                neg_log_det_J -= jnp.sum(self.log_scale_function(s), axis=-1)
                z = BinaryFlip(z, j)

        assert(z.shape[:-1] == neg_log_det_J.shape)
        return z, neg_log_det_J

    def inverse_transform(self, params, x):
        """
        Apply the inverse transformation to the data.

        Parameters:
        params (list): Parameters of the model.
        x (array): Input data.

        Returns:
        tuple: Transformed data and the negative log determinant of the Jacobian.
        """
        neg_log_det_J = jnp.zeros(x.shape[:-1])

        st_list = params

        for i in reversed(range(self.num_transformations)):
            for j in reversed(range(2)):

                x = ReverseBinaryFlip(x, j)

                x_1, x_2 = BinarySplit(x, j)

                s, t = self.apply_net_st(
                                params=st_list[i][j],
                                inputs=x_1)
                # output should be of the shape of x_1
                x = jnp.concatenate([x_1, (x_2 - t)/self.scale_function(s)], axis=-1)
                neg_log_det_J -= jnp.sum(self.log_scale_function(s), axis=-1)
        assert(x.shape[:-1] == neg_log_det_J.shape)

        return x, neg_log_det_J


def test_realnvp_match_mean_inv_cov():
    """
    Test the match_mean_inv_cov method of the RealNVP class.
    """
    key = jax.random.PRNGKey(0)

    dist = RealNVP(10, 32, 2, 0.01, 2)
    params = dist.initial_params()
    zs = jax.vmap(dist.sample, [None, 0])(params, jax.random.split(key, 100000))
    emp_mean = jnp.mean(zs, axis=0)
    emp_cov = jnp.cov(zs.T)

    mean = jnp.array([1.0, 2.0])
    cov = jnp.array([[3.0, -1], [-1, 4.0]])
    inv_cov = jnp.linalg.inv(cov)
    dist2, params2 = dist.match_mean_inv_cov(mean, inv_cov)
    
    zs2 = jax.vmap(dist2.sample, [None, 0])(params2, jax.random.split(key, 100000))
    emp_mean2 = jnp.mean(zs2, axis=0)
    emp_cov2 = jnp.cov(zs2.T)
    
    assert jnp.max(jnp.abs(mean - emp_mean2)) < 0.05
    assert jnp.max(jnp.abs(cov - emp_cov2)) < 0.05
