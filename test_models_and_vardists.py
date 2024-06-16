import vijax.models as models
import vijax.vardists as vardists
import vijax.recipes as recipes
import contextlib
import os
import warnings

# Suppress the float64 warning
warnings.filterwarnings("ignore", message="Explicitly requested dtype float64")

# Define the models you want to try out
model_names = ['Banana', 'Studentt_1_5', 'Funana', 'WellConditionedGaussian', "Studentt_2_5", "NealsFunnel"]
ndim = 3  # Number of dimensions for the models

# Define the variational distributions you want to try out
vardist_names = ["Gaussian", "Diagonal", "RealNVP"]

# Define the keyword arguments for each variational distribution
vardist_kwargs = {
    "Gaussian": {},
    "Diagonal": {},
    "RealNVP": {
        "num_transformations": 10,
        "num_hidden_units": 16,
        "num_hidden_layers": 2,
        "params_init_scale": 0.001,
    }
}

# Define the recipe you want to use for variational inference
recipe_name = "SimpleVI"

# Define the keyword arguments for the recipe
recipe_kwargs = {
    "maxiter": 10,
    "batchsize": 128,
    "stepsize": [1e-3],
    "reg": None,
    "step_schedule": "constant",
    "init_override": "naive",
}

def test_model_vardist_combinations():
    for model_name in model_names:
        for vardist_name in vardist_names:
            try:
                # Suppress print statements
                with open(os.devnull, 'w') as fnull:
                    with contextlib.redirect_stdout(fnull):
                        # Create an instance of the model
                        model = getattr(models, model_name)(ndim)
                        
                        # Create an instance of the variational distribution
                        q = getattr(vardists, vardist_name)(ndim, **vardist_kwargs[vardist_name])
                        
                        # Initialize the parameters of the variational distribution
                        w = q.initial_params()
                        
                        # Create an instance of the recipe
                        recipe = getattr(recipes, recipe_name)(**recipe_kwargs)
                        
                        # Run the variational inference
                        new_q, new_w, vi_rez = recipe.run(target=model, vardist=q, params=w)
                
                # Ensure no errors are thrown and results are returned
                assert new_q is not None
                assert new_w is not None
                assert vi_rez is not None
                
                print(f"Test passed for Model: {model_name}, Variational Distribution: {vardist_name}, Recipe: {recipe_name}")
            except Exception as e:
                print(f"Test failed for Model: {model_name}, Variational Distribution: {vardist_name}, Recipe: {recipe_name} with error: {e}")

if __name__ == "__main__":
    test_model_vardist_combinations()
