# Vijax

`vijax` is a flexible and modular library for variational inference (VI) in Python. The library is designed to be accessible to a wide range of users and applications, enabling them to perform VI without getting tied down by specific abstractions.

The key components of `vijax` are based on a set of abstract base classes and interfaces, which provide a consistent and adaptable structure for building custom VI algorithms. Users can easily extend and modify the library to suit their specific needs, as long as they follow the abstractions provided.

In this document, we present the API design for the main components of `vijax`:

1. [`Model`](#model): Represents a probability model $p(z,x)$, where $z$ are latent variables and $x$ are observed data.
2. [`VarDist`](#variational-distribution): Represents a variational distribution $q_w(z)$.
3. [`Recipe`](#recipe): Provides a high-level interface for running pre-defined VI algorithms.

By adhering to these abstractions, users can easily plug in their own optimization routines, models, and variational distributions, while still benefiting from the core features and utilities provided by the `vijax` library.


## Model

A `Model` is an abstract base class that represents a probability model. 

A concrete implementation of a `Model` must support the following:

* `model = Model(*args,**vargs)` - the constructor can use any argument structure

* `model.ndim`: an integer representing the number of dimensions of the model.

* `model.log_prob(z)`: a required method that evaluates the log-probability of a single `ndim` length vector `z`. 

A concrete implementation of a `Model` can also optionally support:

* `model.sample_prior(PRNGKey)`: a method that samples a single `ndim` length vector from the prior distribution of the model over latent variables in unconstrained space. The method takes a JAX PRNG key as input and returns the sampled vector.

* `model.reference_samples(nsamps)`: a method that samples a set of `nsamps` samples from the posterior / model distribution. This may be implemented by (1) exploiting a special algorithm to exactly sample from the posterior (2) running MCMC (3) looking up samples in a database.

* `model.constrain(z)`: a method that transforms a single given unconstrained latent vector `z` to constrained space. 

<!-- TODO:

* BC suggests calling `sample_prior()` `init()` so as not to limit things to Bayesian models. But not sure what would be the equivalent of sampling from the prior in a non-Bayesian setting.

* Consider adding `PRNGKey` argument to `reference_samples` -->

## VarDist

A `VarDist` is a class that represents a variational distribution $q(z|x)$. A `VarDist` must support the following:

* `vardist = VarDist(ndim, *args, **vargs)` - initialization must take `ndim` as the first argument but is otherwise class dependent. One might provide defaults for all arguments other than `ndim`.

* `vardist.ndim` - the number of dimensions of the unconstrained latent variables.

* `vardist.initial_params()` - initializes the variational parameters (in unconstrained space.)

* `vardist.sample(params, key)` - get a single `ndim` length vector from the variational distribution. Must be differentiable w.r.t. `params`.
  
* `vardist.log_prob(params, z)` - evaluate log probability of a single vector `z`

A concrete instance of `VariationalDistribution` can *optionally* support: 

* `vardist.sample_and_log_prob(params, key)` - generate a sample and evaluate log probability at the same time (can be more efficient and stable for some distributions)

* `vardist.sample_and_log_prob_stl(params, key)` - STL(Sticking the Landing) compatible version of the `vardist.sample_and_log_prob` (*not* expected to be more efficient in general, but should be more numerically stable)

* `vardist.mean_cov(params)` - get the (closed-form) mean and covariance for this variational distribution

* `vardist_new, params = vardist.match_mean_inv_cov(mean,inv_cov)` - get parameters and a new distribution object that match a given mean and inverse covariance

* `vardist_new, params = vardist.match_mean_cov(mean,cov)` - get parameters and a new distribution object that match a given mean and covariance

## Recipe

A `Recipe` is a class that will "do inference". It must support three methods

* `recipe = Recipe(*args,**vargs)` - Initialization can use any argument structure
* `new_vardist, new_params, results = recipe.run(model, vardist, params)` - actually run the recipe
  * The first two return arguments are obvious. The last is any recipe-dependent structure that can contain information about convergence, time used, etc.
* `z = recipe.sample(model, vardist, params, key)` - draw a sample from the recipe for this model and variational distribution
  * In many cases this would just return `vardist.sample(params,key)` but in cases of things like importance weighted objectives, could be more complex.
* `l = recipe.objective(model, vardist, params, key)` - estimate the recipe's objective for a given set of parameters

# Example

Here's an example of how the API might work.

```python
# Get model and variational distribution
model = models.Funana(3)

# Create an instance of the variational distribution
gausssian_q = vardists.Gaussian(3)

# Initialize the parameters of the variational distribution
gaussian_w = gausssian_q.initial_params()

# Create an instance of the recipe
recipe = recipes.SimpleVI(maxiter=10, batchsize=128)

# Run the recipe for variational inference
new_q, new_w, vi_rez = recipe.run(target=model, vardist=gaussian_q, params=gaussian_w)

# Run the recipe for variational inference with a flow variational distribution
flow_q = vardists.RealNVP(
    3,
    num_transformations=6,
    num_hidden_units=16,
    num_hidden_layers=2,
    params_init_scale=0.001
)
# Initialize the parameters of the flow variational distribution
flow_w = flow_q.initial_params()

# Run the same recipe with a flow variational distribution
new_q, new_w, vi_rez = recipe.run(target=model, vardist=flow_q, params=flow_w)


```
## Implemented Models

This repository includes several probabilistic models that can be used for variational inference. Below is a table describing each model along with a reference to their implementation files.

| Model Name                  | Description                                                                                           |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| Well-Conditioned Gaussian   | A Gaussian model with well-conditioned covariance structure. [Reference](vijax/models/InferenceGym.py) |
| Ill-Conditioned Gaussian    | A Gaussian model with ill-conditioned covariance structure. [Reference](vijax/models/InferenceGym.py)  |
| Neal's Funnel               | A model with a funnel-shaped distribution, often used to test sampling algorithms. [Reference](vijax/models/InferenceGym.py) |
| Banana                      | A model with a banana-shaped distribution, used to test inference algorithms. [Reference](vijax/models/InferenceGym.py) |
| Funana                      | A custom model that combines the densities of Neal's Funnel and Banana. [Reference](vijax/models/Funana.py) |
| Studentt-1.5                | A multivariate Student-t distribution with 1.5 degrees of freedom. [Reference](vijax/models/Studentt.py) |
| Studentt-2.5                | A multivariate Student-t distribution with 2.5 degrees of freedom. [Reference](vijax/models/Studentt.py) |

## Implemented Variational Distributions

This repository includes several variational distributions that can be used for variational inference. Below is a table describing each distribution along with a reference to their implementation files.

| Distribution Name           | Description                                                                                           |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| Gaussian                    | A standard Gaussian variational distribution. [Reference](vijax/vardists/Gaussian.py)                 |
| Diagonal Gaussian           | A Gaussian distribution with a diagonal covariance matrix. [Reference](vijax/vardists/Diagonal.py)    |
| RealNVP                     | A flow-based variational distribution using RealNVP transformations. [Reference](vijax/vardists/Flows.py) |


## Requirements

This project requires Python 3.9 or higher. The dependencies for this project are managed using Conda and are listed in the `environment.yml` file. The main libraries used in this project include:

- `jax`: A library for high-performance machine learning research.
- `jaxlib`: The JAX library containing the XLA compiler and other dependencies.
- `numpyro`: A probabilistic programming library built on JAX.
- `tensorflow-probability`: A library for probabilistic reasoning and statistical analysis.
- `inference-gym`: A suite of probabilistic models for benchmarking inference algorithms.

## Setting Up the Environment

To set up the environment, follow these steps:

1. **Install Conda**: If you don't have Conda installed, download and install it.

2. **Create the Conda Environment**: Use the `environment.yml` file to create a new Conda environment. Run the following command in your terminal:

    ```sh
    conda env create -f environment.yml
    ```

3. **Activate the Environment**: Once the environment is created, activate it using:

    ```sh
    conda activate vi_jax
    ```

4. **Run the Tests**: To ensure everything is set up correctly, run the test file `test_models_and_vardists.py`:

    ```sh
    python test_models_and_vardists.py
    ```

This will execute the tests and verify that all models and variational distributions are working as expected with the specified recipes.
