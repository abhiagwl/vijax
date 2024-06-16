import jax
import jax.flatten_util

import jax.numpy as jnp
from functools import partial

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import time
import jax.scipy.optimize
import vijax.utils as utils

class ELBO:
    def __init__(self, target, vardist):
        """
        Initialize the ELBO class with target and variational distribution.

        Parameters:
        - target: The target distribution.
        - vardist: The variational distribution.
        """
        self.target = target
        self.vardist = vardist
        self.ndim = target.ndim
        assert target.ndim == vardist.ndim

    def initial_params(self):
        """
        Get the initial parameters of the variational distribution.

        Returns:
        - Initial parameters of the variational distribution.
        """
        return self.vardist.initial_params()


class ELBO_pq(ELBO):
    def __call__(self, w, key):
        """
        Compute the negative KL divergence from target to variational distribution.

        Parameters:
        - w: Parameters of the variational distribution.
        - key: PRNG key for sampling.

        Returns:
        - ELBO value.
        """
        z = self.target.reference_samples(1, key = key).reshape(-1)
        return self.vardist.log_prob(w, z) - self.target.log_prob(z)

class Joint_ELBO(ELBO):
    def __call__(self, w, key):
        """
        Compute the equivalent of negative symmetric KL divergence with total ELBO.

        Parameters:
        - w: Parameters of the variational distribution.
        - key: PRNG key for sampling.

        Returns:
        - Joint ELBO value.
        Notes: 
            Requires access to reference samples and sample_and_log_prob
        """
        z = self.target.reference_samples(1, key = key).reshape(-1)
        obj_a = self.vardist.log_prob(w, z) - self.target.log_prob(z)

        key, s_key = jax.random.split(key)
        z, lq = self.vardist.sample_and_log_prob(w, key)
        obj_b = self.target.log_prob(z) - lq

        return obj_a + obj_b

class Joint_STLELBO(ELBO):
    def __call__(self, w, key):
        """
        Compute the equivalent of negative symmetric KL divergence with STL ELBO.

        Parameters:
        - w: Parameters of the variational distribution.
        - key: PRNG key for sampling.

        Returns:
        - Joint STLELBO value.
        Notes: 
            Requires access to reference samples
        """
        z = self.target.reference_samples(1, key = key).reshape(-1)
        obj_a = self.vardist.log_prob(w, z) - self.target.log_prob(z)

        z = self.vardist.sample(w, key)
        obj_b = self.target.log_prob(z) - self.vardist.log_prob(jax.lax.stop_gradient(w), z)

        return obj_a + obj_b


class NaiveELBO(ELBO):
    def __call__(self, w, key):
        """
        Compute the naive ELBO.

        Parameters:
        - w: Parameters of the variational distribution.
        - key: PRNG key for sampling.

        Returns:
        - Naive ELBO value.

        Note: Can be less efficient if the variational distribution allows sample_and_log_prob
        """
        z = self.vardist.sample(w, key)
        return self.target.log_prob(z) - self.vardist.log_prob(w, z)


class EntropyELBO(ELBO):
    def __call__(self, w, key):
        """
        Compute the entropy ELBO.

        Parameters:
        - w: Parameters of the variational distribution.
        - key: PRNG key for sampling.

        Returns:
        - Entropy ELBO value.
        Notes: 
            Requires access to entropy
        """
        z = self.vardist.sample(w, key)
        return self.target.log_prob(z) + self.vardist.entropy(w)


class STLELBO(ELBO):
    def __call__(self, w, key):
        """
        Compute the STLELBO.

        Parameters:
        - w: Parameters of the variational distribution.
        - key: PRNG key for sampling.

        Returns:
        - STLELBO value.
        """
        z = self.vardist.sample(w, key)
        return self.target.log_prob(z) - self.vardist.log_prob(jax.lax.stop_gradient(w), z)

class TotalELBO(ELBO):
    def __call__(self, w, key):
        """
        Compute the total ELBO.

        Parameters:
        - w: Parameters of the variational distribution.
        - key: PRNG key for sampling.

        Returns:
        - Total ELBO value.
        Notes: 
            Requires access to sample_and_log_prob
        """
        z, lq = self.vardist.sample_and_log_prob(w, key)
        return self.target.log_prob(z) - lq


def batched(obj, agg_fn = jnp.mean):
    """
    Batch a function to work with an array of keys.

    Parameters:
    - obj: The function to batch.
    - agg_fn: Aggregation function to apply to the batched results.

    Returns:
    - Batched function.
    """
    return lambda w, key: agg_fn(jax.vmap(obj, [None, 0])(w, key))


def ADAM(obj, w0, key, stepsize=0.0001, maxiter=30000, batchsize=1, callback=None, track = False):
    """
    ADAM optimizer as a maximizer.

    Parameters:
    - obj: Objective function to maximize.
    - w0: Initial parameters.
    - key: PRNG key for sampling.
    - stepsize: Step size for the optimizer.
    - maxiter: Maximum number of iterations.
    - batchsize: Batch size for sampling.
    - callback: Optional callback function.
    - track: Whether to track the optimization process.

    Returns:
    - Optimized parameters and the final objective value.
    """
    fun = obj
    w = w0 + 0.0
    ave_f = 0
    m = 0 * w0  # momentum
    v = 0 * w0  # momentum^2
    t0 = time.time()
    
    if utils.is_scalar(stepsize):
        _stepsize = lambda i: stepsize
    elif callable(stepsize):
        _stepsize = stepsize
    else:
        raise ValueError("stepsize must be a scalar or a function")

    callback_list = []

    for i in range(maxiter):
        key1, key = jax.random.split(key)
        key1 = jax.random.split(key1, batchsize)

        f, g = fun(w, key1)

        b1 = 0.9
        b2 = 0.999
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2

        mhat = m / (1 - b1 ** (i + 1))  # bias correction
        vhat = v / (1 - b2 ** (i + 1))  # bias correction

        eps = 1e-8
        w = w + _stepsize(i) * mhat / (eps + jnp.sqrt(vhat))

        alpha = max(0.001, 1 / (i + 1))
        ave_f = (1 - alpha) * ave_f + alpha * f

        if jnp.any(jnp.isnan(w)):
            print("nan encountered, terminating early")
            return w, (ave_f, callback_list)

        if track:
            if callback is not None:
                _mets = callback(i, w)
                _mets["iteration"] = i
                callback_list.append(_mets)
        
    return w, (ave_f, callback_list)

def metric_callback(target, vardist, unflatten, n_samples):
    """
    Create a callback function to compute various metrics during optimization.

    Parameters:
    - target: The target distribution.
    - vardist: The variational distribution.
    - unflatten: Function to unflatten parameters.
    - n_samples: Number of samples for evaluation.

    Returns:
    - Callback function.
    """
    new_metrics = {}
    _met1 = ELBO_pq(target, vardist)
    _batched_met1 = batched(_met1)
    _flat_batched_met1 = lambda w, key: _batched_met1(unflatten(w), key)
    _jitted_flat_batched_met1 = jax.jit(_flat_batched_met1)

    new_metrics["ELBO_pq"] = _jitted_flat_batched_met1

    _met2 = TotalELBO(target, vardist)
    _batched_met2 = batched(_met2)
    _flat_batched_met2 = lambda w, key: _batched_met2(unflatten(w), key)
    _jitted_flat_batched_met2 = jax.jit(_flat_batched_met2)

    new_metrics["TotalELBO"] = _jitted_flat_batched_met2

    # metric 3
    _pos_sampler = lambda w, key: vardist.sample(unflatten(w), key) 
    _batched_samples = jax.vmap(_pos_sampler, in_axes=(None, 0))
    fast_sampler = jax.jit(_batched_samples)
    reference_samples = target.reference_samples(n_samples, key = jax.random.PRNGKey(0))

    _met3 = lambda w, keys: jax.jit(utils.wass1)(
        reference_samples,
        fast_sampler(w, keys),
    )
    new_metrics["Wasserstein1"] = _met3

    def callback(i, w):
        _mets = {
            name: func(w, jax.random.split(jax.random.PRNGKey(i), n_samples)) 
            for name, func in new_metrics.items()
        }
        return _mets
    return callback

def constant_schedule(iter, step):
    """
    Constant step size schedule.

    Parameters:
    - iter: Current iteration.
    - step: Step size.

    Returns:
    - Constant step size.
    """
    return step

def decay_schedule(iter, step, maxiter):
    """
    Decaying step size schedule.

    Parameters:
    - iter: Current iteration.
    - step: Initial step size.
    - maxiter: Maximum number of iterations.

    Returns:
    - Decayed step size.
    """
    if iter < maxiter//2:
        return step
    elif (iter >= maxiter//2) and (iter < 3*maxiter//4):
        return step/2
    else:
        return step/4


def get_step_schedule(step, step_schedule, maxiter):
    """
    Get the step size schedule function.

    Parameters:
    - step: Initial step size.
    - step_schedule: Type of step size schedule ("constant" or "decay").
    - maxiter: Maximum number of iterations.

    Returns:
    - Step size schedule function.
    """
    if step_schedule == "constant":
        return partial(constant_schedule, step = step)
    elif step_schedule == "decay":
        return partial(decay_schedule, step = step, maxiter = maxiter)
    else:
        raise Exception(f"Unknown step schedule {step_schedule}")

def run_opt(
        stepsize, w0, maxiter, key, obj, batchsize, stepschedule,
        callback = None, 
        eval_obj = None, 
        eval_N = int(2**10), 
        track = False, 
        optimizer = ADAM,
    ):
    """
    Run the optimization process.

    Parameters:
    - stepsize: Initial step size.
    - w0: Initial parameters.
    - maxiter: Maximum number of iterations.
    - key: PRNG key for sampling.
    - obj: Objective function.
    - batchsize: Batch size for sampling.
    - stepschedule: Step size schedule.
    - callback: Optional callback function.
    - eval_obj: Evaluation objective function.
    - eval_N: Number of samples for evaluation.
    - track: Whether to track the optimization process.
    - optimizer: Optimizer function.

    Returns:
    - Optimized parameters and the final objective value.
    """
    key1, key2 = jax.random.split(key)
    try:
        # Do initial opt
        scheduled_steps = get_step_schedule(stepsize, stepschedule, maxiter=maxiter)
        w, (_, callback_list) = optimizer(
            obj = obj,
            w0 = w0,
            key = key1,
            stepsize = scheduled_steps,
            maxiter = maxiter,
            batchsize = batchsize,
            callback = callback,
            track = track
        )

        # Re-estimate elbo with final parameters
        ave_f = eval_obj(w, jax.random.split(key2, eval_N))
        return w, (ave_f, callback_list)
    except Exception as e:
        print(f"Exception for stepsize {stepsize}: {e}")
        return None, (-jnp.inf, [{}])

def init_z(target):
    """
    Initialize the latent variable z.

    Parameters:
    - target: The target distribution.

    Returns:
    - Dictionary of initial z values.
    """
    z0_c1 = jnp.zeros(target.ndim)
    z0_c2 = jax.random.normal(jax.random.PRNGKey(0), (target.ndim,))
    try:
        u = jax.jit(lambda zs: jax.vmap(target.sample_prior)(zs))(
                jax.random.split(jax.random.PRNGKey(0), 100)
            )
        z0_c3 = jnp.median(u, axis=0)
        return {"zero": z0_c1, "random": z0_c2, "prior_median": z0_c3}
    except (AttributeError, NotImplementedError):
        return {"zero": z0_c1, "random": z0_c2, "prior_median": None}
    except Exception as e:
        raise e

def laplace(z0, target, vardist):
    """
    Perform Laplace approximation.

    Parameters:
    - z0: Initial z value.
    - target: The target distribution.
    - vardist: The variational distribution.

    Returns:
    - Updated variational distribution and parameters.
    """
    @jax.jit
    def neg_log_prob(z):
        logp = target.log_prob(z)
        return jax.lax.select(jnp.isnan(logp),-jnp.inf,-logp)
    results = jax.scipy.optimize.minimize(neg_log_prob, z0, method="BFGS")
    zstar = results.x
    invH = jnp.linalg.inv(jax.hessian(neg_log_prob)(zstar))
    try:
        new_vardist, new_params = vardist.match_mean_cov(zstar, invH)
        return new_vardist, new_params
    except (AttributeError, NotImplementedError) as e:
        print(
            f"WARNING: match_mean_cov not implemented for {vardist.__class__.__name__}. Returning the same variational distribution when match_mean_cov is called in laplace."
        )
        print(e)
        return vardist, (zstar, invH)
    except Exception as e:
        raise e

def map(z0, target, vardist):
    """
    Perform MAP estimation.

    Parameters:
    - z0: Initial z value.
    - target: The target distribution.
    - vardist: The variational distribution.

    Returns:
    - Updated variational distribution and parameters.
    """
    @jax.jit
    def neg_log_prob(z):
        logp = target.log_prob(z)
        return jax.lax.select(jnp.isnan(logp),-jnp.inf,-logp)
    results = jax.scipy.optimize.minimize(neg_log_prob, z0, method="BFGS")
    zstar = results.x
    try:
        new_vardist, new_params = vardist.match_mean_cov(zstar, jnp.eye(target.ndim))
        return new_vardist, new_params
    except (AttributeError, NotImplementedError) as e:
        print(
            f"WARNING: match_mean_cov not implemented for {vardist.__class__.__name__}. Returning the same variational distribution when match_mean_cov is called in map."
        )
        print(e)
        return vardist, (zstar, jnp.eye(target.ndim))
    except Exception as e:
        raise e




def find_initial_base_dist_params(initial_params, eval_obj, target, vardist, key, eval_N = 1000, override = None):
    """
    Find the initial parameters for the base distribution.

    Parameters:
    - initial_params: Initial parameters.
    - eval_obj: Evaluation objective function.
    - target: The target distribution.
    - vardist: The variational distribution.
    - key: PRNG key for sampling.
    - eval_N: Number of samples for evaluation.
    - override: Override method for initialization.

    Returns:
    - Updated variational distribution, parameters, ztype, and initialization type.
    """

    if override is not None:
        if override == "naive":
            return vardist, initial_params, "N/A", "naive"
    key, skey = jax.random.split(key)
    # take naive parameters
    try:
        params_naive = initial_params
        vardist_naive = vardist
        key, skey = jax.random.split(skey)
        naive_obj = jnp.mean(jax.vmap(eval_obj(target, vardist_naive), [None, 0])(params_naive, jax.random.split(key, eval_N)))
        ztype_naive = "N/A"
    except AttributeError as e:
        naive_obj = jnp.array(-jnp.inf)

    # take map parameters
    try:
        z0s = init_z(target)
        best_vardist = None
        best_obj = -jnp.inf
        best_params = None
        best_ztype = None
        for z0_type, z0 in z0s.items():
            print(f"Trying {z0_type} initialisation", end="\r")
            vardist_map, params_map = map(z0, target, vardist)
            key, skey = jax.random.split(skey)
            map_obj = jnp.mean(
                jax.vmap(eval_obj(target, vardist_map), [None, 0])(params_map, jax.random.split(key, eval_N))
            )
            if map_obj > best_obj:
                best_vardist = vardist_map
                best_obj = map_obj
                best_params = params_map
                best_ztype = z0_type
        map_obj = best_obj
        vardist_map = best_vardist
        params_map = best_params
        ztype_map = best_ztype

    except AttributeError:
        map_obj = jnp.array(-jnp.inf)
    except Exception as e:
        raise e
    # exit()
    # take laplace parameters
    try:
        z0s = init_z(target)
        best_vardist = None
        best_obj = -jnp.inf
        best_params = None
        best_ztype = None
        for z0_type, z0 in z0s.items():
            print(f"Trying {z0_type} initialisation", end="\r")
            vardist_laplace, params_laplace = laplace(z0, target, vardist)
            key, skey = jax.random.split(skey)
            laplace_obj = jnp.mean(
                jax.vmap(eval_obj(target, vardist_laplace), [None, 0])(params_laplace, jax.random.split(key, eval_N))
            )
            if laplace_obj > best_obj:
                best_vardist = vardist_laplace
                best_obj = laplace_obj
                best_params = params_laplace
                best_ztype = z0_type
        laplace_obj = best_obj
        vardist_laplace = best_vardist
        params_laplace = best_params
        ztype_laplace = best_ztype
    except AttributeError as e:
        laplace_obj = jnp.array(-jnp.inf)
        raise e
    except Exception as e:
        raise e

    print("\nInitial ELBOS")
    utils.print_variables(
        {
            "Naive": naive_obj,
            "MAP": map_obj,
            "Laplace": laplace_obj,
        }
    )

    winner = jnp.nanargmax(jnp.array([naive_obj, map_obj, laplace_obj]))
    if override is not None:
        if override == "naive":
            winner = 0
        elif override == "map":
            winner = 1
        elif override == "laplace":
            winner = 2
        else:
            raise Exception("Unknown override")
        

    if winner == 0:  
        vardist = vardist_naive
        params = params_naive
        ztype = ztype_naive
        init_type = "naive"
    elif winner == 1:
        vardist = vardist_map
        params = params_map
        ztype = ztype_map
        init_type = "map"
    elif winner == 2:
        vardist = vardist_laplace
        params = params_laplace
        ztype = ztype_laplace
        init_type = "laplace"
    else:
        raise Exception("Unknown winner")
    return vardist, params, ztype, init_type

class SimpleVI:
    def __init__(
            self, 
            maxiter=1000, 
            batchsize=128, 
            stepsize=[3e-3], 
            reg=None, 
            step_schedule="constant",
            elbo=STLELBO, 
            eval_elbo=TotalELBO, 
            init_override=None, 
            optimizer=ADAM,
        ):
        """
        Initialize the SimpleVI class with the given parameters.

        Parameters:
        - maxiter (int): Maximum number of iterations for optimization.
        - batchsize (int): Batch size for training.
        - stepsize (List[float]): List of step sizes for the optimizer.
        - reg (float, optional): Regularization parameter.
        - step_schedule (str): Schedule for step size adjustment. Can be "constant" or "decay".
        - elbo (Type[ELBO]): ELBO class for training.
        - eval_elbo (Type[ELBO]): ELBO class for evaluation.
        - init_override (Optional[str]): Override for initialization method. Can be "naive" or None.
        - optimizer (Type[Optimizer]): Optimizer to be used for training.
        """
        self.maxiter = maxiter
        self.batchsize = batchsize
        self.reg = reg
        self.ELBO = elbo
        self.stepsizes = stepsize
        self.EvalELBO = eval_elbo
        self.init_override = init_override
        self.step_schedule = step_schedule
        self.optimizer = optimizer

    def sample(self, target, vardist, params, key):
        """
        Sample from the variational distribution.

        Parameters:
        - target (Distribution): Target distribution.
        - vardist (VariationalDistribution): Variational distribution.
        - params (PyTree): Parameters of the variational distribution.
        - key (jax.random.PRNGKey): PRNG key for sampling.

        Returns:
        - Array: Sampled values from the variational distribution.
        """
        return vardist.sample(params, key)

    def objective(
        self,
        target,
        vardist,
        params,
        key,
    ):
        """
        Compute the objective function (ELBO).

        Parameters:
        - target (Distribution): Target distribution.
        - vardist (VariationalDistribution): Variational distribution.
        - params (PyTree): Parameters of the variational distribution.
        - key (jax.random.PRNGKey): PRNG key for evaluation.

        Returns:
        - float: ELBO value.
        """
        # NOTE: Is this efficient? Should we return a function? 
        return self.EvalELBO(target, vardist)(params, key)
    
    def run(self, target, vardist, params, seed=2):
        """
        Run the variational inference optimization.

        Parameters:
        - target (Distribution): Target distribution.
        - vardist (VariationalDistribution): Variational distribution.
        - params (PyTree): Initial parameters of the variational distribution.
        - seed (int): Random seed for reproducibility.

        Returns:
        - vardist (VariationalDistribution): Optimized variational distribution.
        - params (PyTree): Optimized parameters.
        - elbo (float): Final ELBO value.
        - ztype (str): Type of initialization used.
        - init_type (str): Initialization method used.
        - best_step (float): Best step size found.
        - callback_list (List): List of callback values during optimization.
        """
        skey = jax.random.PRNGKey(seed)

        # Get the initial base distribution parameters
        key, skey = jax.random.split(skey)
        vardist, params, ztype, init_type = find_initial_base_dist_params(
            initial_params=params,
            eval_obj= self.EvalELBO,
            target = target,
            vardist = vardist,
            key = key,
            eval_N = 1000,
            override = self.init_override
        )

        print(f"Selected {init_type} initialization for {vardist.__class__.__name__} on {target.__class__.__name__} with ndim: {target.ndim}")
        print(f"Initial ztype: {ztype}")

        # Create the training objective
        # TODO: Make creating an objective a function
        w0, unflatten = jax.flatten_util.ravel_pytree(params)
        if self.reg is not None:
            assert utils.is_scalar(self.reg)

        with timeit("Creating objective", False):
            # Create the training objective
            train_elbo = self.ELBO(target, vardist)
            # Batch it
            batched_train_elbo = batched(train_elbo)
            # Flatten it
            _flat_batched_train_elbo = lambda w, key: batched_train_elbo(unflatten(w), key)
            # Add regularizer
            if self.reg is not None:
                flat_batched_train_elbo = lambda w, key: _flat_batched_train_elbo(w, key) - self.reg * jnp.sum(w**2)
            else:
                flat_batched_train_elbo = _flat_batched_train_elbo
            # Take grad
            _flat_batched_train_elbo_and_grad = jax.value_and_grad(flat_batched_train_elbo, 0)
            # JIT
            flat_batched_train_elbo_and_grad = jax.jit(_flat_batched_train_elbo_and_grad)

            # Create the evaluation objective
            eval_elbo = self.EvalELBO(target, vardist)
            # Batch it to accept a batch of keys. 
            _batched_eval_elbo = batched(eval_elbo)
            # Flatten it
            _flat_batched_eval_elbo = lambda w, keys: _batched_eval_elbo(unflatten(w), keys)
            # JIT
            flat_batched_eval_elbo = jax.jit(_flat_batched_eval_elbo)

        # Select the batchsize
        if utils.is_scalar(self.batchsize):
            batchsize = self.batchsize
        else:
            raise Exception(f"Unknown batchsize. Expected int, got {self.batchsize}")

        # Make test call to compile
        with timeit("Test call", True):
            key, skey = jax.random.split(skey)
            flat_batched_train_elbo_and_grad(w0, jax.random.split(key, batchsize))

        # Start the optimization process
        t0 = time.time()
        key, skey = jax.random.split(skey)
        _run_opt = partial(run_opt, 
            w0 = w0,
            obj = flat_batched_train_elbo_and_grad,
            maxiter= self.maxiter,
            key = key,
            batchsize = batchsize,
            stepschedule = self.step_schedule,
            callback= None,
            eval_obj = flat_batched_eval_elbo,
            eval_N = int(max(batchsize, 2**17)),
            track = True, 
            optimizer = self.optimizer
        )
        ws = {}
        elbos = {}
        callback_lists = {}
        for step in self.stepsizes:
            print(f"Working with step_size {step}...")
            w, (elbo, callback_list) = _run_opt(stepsize= step)
            ws[step] = w
            elbos[step] = elbo
            callback_lists[step] = callback_list
        best_step = max(elbos, key=elbos.get)
        w = ws[best_step]
        elbo = elbos[best_step]
        callback_list = callback_lists[best_step]

        print(f"{'iters':<10}{'beststep':<10}{'elbo':<10}{'time':<10}")

        print(
            f"{self.maxiter:<10}{best_step:<10.4g}{elbo:<10.4g}{time.time()-t0:<10.4g} sec"
        )
        return vardist, unflatten(w), (elbo, ztype, init_type, best_step, callback_list)


class timeit:
    def __init__(self, desc="timing", show=True):
        """
        Initialize the timeit context manager.

        Parameters:
        - desc (str): Description of the timed block.
        - show (bool): Whether to print the timing information.
        """
        self.desc = desc
        self.show = show

    def __enter__(self):
        """
        Enter the context manager, start the timer.
        """
        self.t0 = time.time()
        if self.show:
            print(self.desc + " ... ", end="", flush=True)

    def __exit__(self, *args):
        """
        Exit the context manager, stop the timer and print the elapsed time.
        """
        if self.show:
            print(f"Time taken: {time.time()-self.t0:.4g} sec")
