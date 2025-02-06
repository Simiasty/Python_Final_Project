from setuptools import setup, find_packages

setup(
    name="python_final_project",            # Name of your project
    version="1.0",                          # Version of your project
    description="A Python project for analyzing linguistic data.",  # Short description
    packages=find_packages(),               # Automatically find all packages (folders with __init__.py)
    install_requires=[                      # List of dependencies
        "numpy",
        "pandas",
        "matplotlib",
        "pytest",
    ],
    python_requires=">=3.6",                # Specify the Python versions supported
)
