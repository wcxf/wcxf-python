from setuptools import setup, find_packages

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="wcxf",
    version="2.0",
    author="David M. Straub, Jason Aebischer",
    author_email="david.straub@tum.de, jason.aebischer@tum.de",
    license="MIT",
    url="https://wcxf.github.io",
    description="Python API and command line interface for the Wilson Coefficient exchange format",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["wilson"],
)
