# Import necessary libraries and modules
import vijax.vardists as vardists 
import vijax.recipes as recipes 
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
from numpyro.infer import MCMC, NUTS
from jax import flatten_util
import matplotlib.pyplot as plt
import joblib

mem = joblib.Memory('./cache')

# Define a multivariate model in numpyro
def model(dim=10):
    y = numpyro.sample("y", dist.Normal(0, 3))
    numpyro.sample("x", dist.Normal(jnp.zeros(dim - 1), jnp.exp(y / 2)))

# Set the dimension of the model and generate some data
dim = 2

# Define a wrapper class for the Numpyro model
class NumpyroModelWrapper:
    def __init__(self, model, key, model_args, model_kwargs):
        self.model_utils = initialize_model(key, model, model_args=model_args, model_kwargs=model_kwargs)
        self.init_z, self.unflatten = flatten_util.ravel_pytree(self.model_utils.param_info.z)

    @property
    def ndim(self):
        return self.init_z.shape[-1]

    @ndim.setter
    def ndim(self, ndim):
        raise ValueError("Cannot set ndim.")

    def log_prob(self, z):
        return -self.model_utils.potential_fn(self.unflatten(z))
    
    def constrain(self, z):
        return flatten_util.ravel_pytree(self.model_utils.postprocess_fn(self.unflatten(z)))[0]

# Initialize the Numpyro model wrapper
vijax_model = NumpyroModelWrapper(model, jax.random.PRNGKey(0), (dim,), {})
vijax_model.ndim

# Define the variational distribution and inference recipe
def run_vi(vardist_class, *vardist_args):
    q = vardist_class(*vardist_args)
    w = q.initial_params()
    recipe = recipes.SimpleVI(maxiter=10000, batchsize=128, stepsize=[1e-3], step_schedule="constant", init_override="naive")
    vi_rez = mem.cache(recipe.run)(target=vijax_model, vardist=q, params=w)
    print(f"ELBO: {vi_rez[2][0]:.4f}")
    vi_sampler = jax.jit(jax.vmap(vi_rez[0].sample, in_axes=(None, 0)))
    vi_samples = vi_sampler(vi_rez[1], jax.random.split(jax.random.PRNGKey(2), 128*1000))
    vi_samples = jax.jit(jax.vmap(vijax_model.constrain, in_axes=(0)))(vi_samples)
    return vi_samples

# Run variational inference for different VI families
vi_samples_diag = run_vi(vardists.Diagonal, vijax_model.ndim)
vi_samples_gauss = run_vi(vardists.Gaussian, vijax_model.ndim)
vi_samples_realnvp = run_vi(vardists.RealNVP, vijax_model.ndim, 10, 8, 2, 0.001)

# Run NUTS (No-U-Turn Sampler) on the model
nuts_kernel = NUTS(model, adapt_mass_matrix=True, adapt_step_size=True)
mcmc = MCMC(nuts_kernel, num_chains=128, num_warmup=1000, num_samples=1000, thinning=10, progress_bar=False)
mcmc.run(jax.random.PRNGKey(1), dim)
mcmc_samples = mcmc.get_samples()

# Function to plot pairwise marginals
def plot_pairwise(samples_list, titles):
    num_vars = samples_list[0].shape[1]
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(12, 12))
    for i in range(num_vars):
        for j in range(num_vars):
            if i != j:
                for samples, title in zip(samples_list, titles):
                    axes[i, j].scatter(samples[:, i], samples[:, j], alpha=0.3, label=title)
                if i == num_vars - 1:
                    axes[i, j].set_xlabel(f'Var {j}')
                if j == 0:
                    axes[i, j].set_ylabel(f'Var {i}')
            else:
                axes[i, j].axis('off')
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

# Collapse the samples dictionary into an array
mcmc_samples_array = jnp.concatenate(
    [mcmc_samples[key][..., None] if mcmc_samples[key].ndim == 1 else mcmc_samples[key] for key in mcmc_samples.keys()],
    axis=1
)
# Plot pairwise marginals for NUTS and VI samples
plot_pairwise([mcmc_samples_array, vi_samples_diag, vi_samples_gauss, vi_samples_realnvp], 
              ['NUTS', 'Diagonal', 'Gaussian', 'RealNVP'])

# Function to plot violin plots
def plot_violin(samples_list, titles):
    num_vars = samples_list[0].shape[1]
    fig, axes = plt.subplots(1, num_vars, figsize=(4 * num_vars, 6))
    for i in range(num_vars):
        for samples, title in zip(samples_list, titles):
            axes[i].violinplot(samples[:, i], positions=[titles.index(title) + 1], showmeans=True)
        axes[i].set_xticks(range(1, len(titles) + 1))
        axes[i].set_xticklabels(titles)
        axes[i].set_title(f'Var {i}')
    plt.tight_layout()
    plt.show()

# Plot violin plots for NUTS and VI samples
plot_violin([mcmc_samples_array, vi_samples_diag, vi_samples_gauss, vi_samples_realnvp], 
            ['NUTS', 'Diagonal', 'Gaussian', 'RealNVP'])

# Calculate and print mean and variance for VI and MCMC samples
def calculate_statistics(vi_samples, mcmc_samples_array, vi_name):
    vi_mean, vi_variance = jnp.mean(vi_samples, axis=0), jnp.var(vi_samples, axis=0)
    mcmc_mean, mcmc_variance = jnp.mean(mcmc_samples_array, axis=0), jnp.var(mcmc_samples_array, axis=0)
    mean_difference, variance_difference = vi_mean - mcmc_mean, vi_variance - mcmc_variance

    print(f"Statistics for {vi_name} VI:")
    print(f"{'VI Mean:':<20} {vi_mean}")
    print(f"{'MCMC Mean:':<20} {mcmc_mean}")
    print(f"{'Mean Difference:':<20} {mean_difference}")
    print(f"{'VI Variance:':<20} {vi_variance}")
    print(f"{'MCMC Variance:':<20} {mcmc_variance}")
    print(f"{'Variance Difference:':<20} {variance_difference}")
    print("\n" + "-"*50 + "\n")

# Print statistics for different VI families
calculate_statistics(vi_samples_diag, mcmc_samples_array, "Diagonal")
calculate_statistics(vi_samples_gauss, mcmc_samples_array, "Gaussian")
calculate_statistics(vi_samples_realnvp, mcmc_samples_array, "RealNVP")

