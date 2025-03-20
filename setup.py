import setuptools

from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "bainite_boundaries",
    version = "0.0.1",
    author = "Bernd Schuscha",
    author_email = "bernd.schuscha@mcl.at",
    description = "Enter a short description for your package here",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://gitlab.mcl.at/b.schuscha/bainite_boundaries.git",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: I don't know yet",
        "Operating System :: OS Independent",
    ],


    packages=find_packages(),         # Automatically finds packages in the root directory
    package_dir={"": "."},            # Sets the package root to the current directory
    python_requires=">=3.9"
    #package_dir = {"": "bainite_boundaries"},
    #packages = setuptools.find_packages(where="bainite_boundaries"),
    #python_requires = ">=3.9"
)
