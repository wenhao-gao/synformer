# from setuptools import setup

# setup(packages=["synformer"])
from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

# get __version__ from _version.py
ver_file = path.join("synformer", "version.py")
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))


# Check if README.md exists and read its content
long_description = ""
if path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Long description not available."

# read the contents of requirements.txt
with open(path.join(this_directory, "env.txt"),
          encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="synformer",
    version=__version__,
    license="Apache-2.0",
    description="Synformer: Generative Model for Synthesizable Molecule Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wenhao-gao/synformer",
    author="Wenhao Gao, Shitong Luo, Connor W. Coley",
    author_email="gaowh19@gmail.com",
    packages=find_packages(exclude=["test"]),
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    setup_requires=["setuptools>=38.6.0"],
)