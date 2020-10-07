import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="psychsim",
    version="1.0",
    author="David Pynadath",
    author_email="pynadath@ict.usc.edu",
    description="PsychSim is an open-source social-simulation framework using decision-theoretic agents who have a theory of mind",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usc-psychsim/psychsim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
