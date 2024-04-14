from setuptools import setup, find_packages

with open("./README.md", "r") as f:
    long_description = f.read()

setup(
    name="universe_render",
    version="0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="Rendering the universe in hydro sims",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pointeee/universe_render",
    packages=find_packages(), 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'numpy-quaternion', 
        'numba',
        'mpi4py'
    ],
)