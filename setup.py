from setuptools import setup, find_packages
setup(
    name="hmsa",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "networkx", "numpy", "matplotlib", "scikit-learn", "numba", "pandas", "scipy"
    ],
)
