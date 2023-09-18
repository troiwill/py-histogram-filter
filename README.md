# Histogram Filter for Python

This package provides a base class and tools for implementing a Bayes Histogram Filter. Users would need to implement
the `histogram.predict(...)` and `histogram.update(...)` methods.

This code was inspired by [this code](https://irwinsnet.github.io/probrob/nb07.html).

### How To Set Up
Run the following commands to install this package. **Note**: This package was developed using Python 3.8.
```commandline
# Install the dependencies.
pip install numpy==1.24.4

# Install this package.
mkdir -p ~/repos && cd ~/repos \
    && git clone https://github.com/troiwill/py-histogram-filter.git \
    && cd py-histogram-filter \
    && python -m build \
    && pip install dist/*.whl
```

### Running Unittests
There are a set of test scripts in the `tests` directory. Run them to ensure that the package works
(according to the tests available).
```commandline
python -m unittest discover tests
```

If anything is broken (one or more tests fail), please submit a
[GitHub Issue](https://github.com/troiwill/py-histogram-filter/issues).

### Contact
Please submit a [GitHub Issue](https://github.com/troiwill/py-histogram-filter/issues) if you have any questions, 
concerns, or suggestions regarding this package.
