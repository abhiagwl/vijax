from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vijax",  # Replace with your own package name
    version="0.0.3",
    author="Abhinav Agrawal",
    author_email="agrawal.abhinav1@gmail.com",
    description="A JAX library to run VI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhiagwl/vijax",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "jax",
        "jaxlib",
        "numpyro",
        "tensorflow-probability",
        "inference-gym"
    ],
)
