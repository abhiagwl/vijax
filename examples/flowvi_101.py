from vijax import models, vardists, recipes
import vijax.recipes as recipes

# Define a single model and variational distribution
ndim = 3  # Number of dimensions for the model

# Create an instance of the model
model = models.Funana(ndim)

# Create an instance of the variational distribution
q = vardists.RealNVP(
    ndim,
    num_transformations=10,
    num_hidden_units=16,
    num_hidden_layers=2,
    params_init_scale=0.001
)

# Initialize the parameters of the variational distribution
w = q.initial_params()

# Create an instance of the recipe
recipe = recipes.SimpleVI(
    maxiter=10,
    batchsize=128,
    stepsize=[1e-3],
    reg=None,
    step_schedule="constant",
    init_override="naive"
)

# Run the variational inference
new_q, new_w, vi_rez = recipe.run(target=model, vardist=q, params=w)

# Print the results
print(f"Model: Funana, Variational Distribution: RealNVP, Recipe: SimpleVI")
print(f"ELBO: {vi_rez[0]:.4f}")


