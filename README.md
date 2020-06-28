# PsychSim
Decision-theoretic agents with Theory of Mind for social simulation.

# Documentation
The latests documentation can be found [here][3].

# Installation/Development
In order to install the module for development (see pip [docs][1]):

1. clone the repository: `git clone https://github.com/usc-psychsim/psychsim.git`
2. `cd` into the repository directory and use: `pip install -e .`
3. remove later if desired using: `pip uninstall psychsim`

Using the `-e` argument will allow you to actively work on the package locally without needing to re-install every time you make a modification to the source. In order to install "permanently" simply omit the `-e` in step 2 above.

To install from a `requrements.txt` (i.e. as part of an automated docker build) [use][2]:
```
-e git+https://github.com/usc-psychsim/psychsim.git#egg=psychsim
```

[1]: https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs
[2]: https://stackoverflow.com/questions/16584552/how-to-state-in-requirements-txt-a-direct-github-source
[3]: https://psychsim.readthedocs.io/en/latest/
