import vijax.models as models
import vijax.vardists as vardists
import vijax.recipes as recipes

# Get model and variational distribution
model = models.Funana(3)

# Create an instance of the variational distribution
gaussian_q = vardists.Gaussian(3)

# Initialize the parameters of the variational distribution
gaussian_w = gaussian_q.initial_params()

# Create an instance of the recipe
recipe = recipes.SimpleVI(maxiter=10, batchsize=128)

# Run the recipe for variational inference
new_q, new_w, vi_rez = recipe.run(target=model, vardist=gaussian_q, params=gaussian_w)

# Print the results for Gaussian variational distribution
print(f"Model: Funana, Variational Distribution: Gaussian, Recipe: SimpleVI")
print(f"ELBO: {vi_rez[0]:.4f}")

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

# Print the results for RealNVP variational distribution
print(f"Model: Funana, Variational Distribution: RealNVP, Recipe: SimpleVI")
print(f"ELBO: {vi_rez[0]:.4f}")
