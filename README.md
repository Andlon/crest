# crest

*crest* is an experimental C++14 library for solving the wave equation with finite element methods in non-convex domains.
 It accompanies my Master's thesis *Finite element solutions to the wave
                                    equation in non-convex domains: A relaxation of the CFL condition in the
                                    presence of local mesh refinement*.

crest is a library, but it is in no way ready for use in real-world projects. I must emphasize its experimental nature.
In particular, like most experimental prototypes, it suffers from:

- Leaky abstractions at the wrong level.
- Sparse documentation.
- The occasional hack or unsafe code that you just should not do in production code.

In terms of the last point, I'm thinking particularly about how the library in several places stores pointers
to objects passed to its constructor, and thus implicitly requires the passed object to outlive the receiving
object. In modern production-quality code, you would prefer to use e.g. shared pointers for this,
but the unsafe approach of simply storing pointers was much quicker to implement 
when integrating several libraries together.

With that said, a lot of  work has gone into *crest*, and it does solve some problems which are
 very much non-trivial to solve. In the case that anyone is interested in the topic of
my thesis, I have provided this library under a permissive open-source license, so that they do not have
to start from scratch.

The following is a summary of the features provided by the library,
referencing relevant portions of my thesis:

- A header-only template library that abstracts over the scalar type (i.e. double,
float).
- Geometry:
    - A data structure for 2D triangle meshes.
    - A fast and memory-efficient implementation of the newest vertex bisection
(NVB) algorithm, as well as a fast implementation of the Threshold algorithm
from Lemma 4.2.1.
    - A BiscaleMesh data structure which serves as an abstraction over a coarse mesh and its refinement.
    Among other things, the data structure provides a
convenient way to quickly determine fine triangles that are descendants
of specific triangles in the coarse mesh.
- Corrector computation:
    - An abstraction over different ways to compute correctors, which simplifies
experimenting with new ways to compute correctors.
    - A default implementation based on direct solution with Sparse LU from the
[Eigen](http://eigen.tuxfamily.org/) library.
    - An implementation which leverages the Sparse Cholesky decomposition from
Eigen in conjunction with the Schur complement reduction method based
on the Conjugate Gradient method described in 6.3.1.
    - An implementation based on AMG block-preconditioned GMRES as described in 6.3.2. The AMG functionality
    as well as the LGMRES solver used are powered by the open-source library
    [amgcl](https://github.com/ddemidov/amgcl).
- A comprehensive test suite consisting of unit tests, automatic convergence tests
and property-based tests with randomized input ensure correctness of many of
the algorithms that have been implemented. In particular, the output from the
different implementations for corrector computation is verified to satisfy fundamental properties of the method for randomized input meshes.
- Fast quadrature computation for polynomial orders ranging from 1 to 20,
using compile-time selection of quadrature strength.
- Temporal discretization schemes: Leapfrog, mass-lumped Leapfrog, Crank-Nicolson.
- A limited abstraction for solving the wave equation for various input data,
including functionality to estimate the error with respect to some reference solution.

*crest* has only been tested on Linux systems, and only with the GCC compiler.

Finally, I want to note that anyone who wants to experiment or use crest for any purpose is welcome to
contact me with questions (for example by making an issue on GitHub or sending me an email if you can dig up an address).
Because it's unlikely that even a single person will ever look at the code,
I can not justify the time for writing extensive documentation.


## Getting started

*crest* is a header-only library, which means that you can simply include whatever header files
you want to use from `include/crest/` in your project. Usually that means you want to add the `include/`
folder to your list of include directories. Note that *crest* requires C++14 support in your compiler.

#### Dependencies

There are a number of dependencies associated with crest. Depending on which header files you use,
the dependencies you actually need may vary. In addition, if you wish to run the test suite of *crest*,
there are additional dependencies. The list of library dependencies is as follows:

- [Eigen](http://eigen.tuxfamily.org/), version >= 3.3.1.
- [Boost](http://www.boost.org/), version >= 1.55. Earlier versions *may* work. On Ubuntu-based systems,
   it should be sufficient to install `libboost-all-dev`.
- [amgcl](https://github.com/ddemidov/amgcl). This is only needed if you want to use the
   AMG/GMRES-based corrector solver. To determine an exact working version,
   check out the version referenced by the amgcl git submodule used by this repository.
- [HDF5](https://support.hdfgroup.org/HDF5). This is only necessary for using the functionality in the
   `include/io` module, which enables saving and loading correctors to/from disk. On Ubuntu-based systems,
   I think `libhdf5-dev` should be sufficient, but perhaps `libhdf5-cpp-11` is also required.

For running the test suite of *crest*, the list of dependencies is as follows:

- [cmake](https://cmake.org/), version >= 3.2. This is the build system used by *crest*.
- [Google Test](https://github.com/google/googletest). Powers the test suite.
- [JSON for Modern C++](https://github.com/nlohmann/json). This is only necessary for the
  *experiment runner* (more on that later), and not for the test suite itself.
- [RapidCheck](https://github.com/emil-e/rapidcheck). Many of the tests in the test suite are property-based tests,
   which means that they ensure that the implementation satisfies certain properties for randomized input.
   
#### Compiling the test suite

First, you need to clone the repository:

```
$ cd <wherever-you-want>
$ git clone https://github.com/Andlon/crest.git
$ cd crest
```

You should now be at the root directory of the repository.

When working directly with *crest*, several of its dependencies are automatically included by submodules. Others
must be installed separately. On recent Ubuntu-based systems, you may need the following:

```
$ sudo apt-get install libboost-all-dev libhdf5-dev libhdf5-cpp-11 cmake
```

You will also need to make the [Eigen](http://eigen.tuxfamily.org/) library available for inclusion. Note that the
version of Eigen included in the Ubuntu packages for Eigen may be too old. To make this process simpler,
the library allows to use a local version of Eigen by placing the `Eigen/` headers into `external/include/Eigen/`.
For example, you may do the following (from the root directory of this repository) to locally install Eigen 3.3.2:

```
$ wget -q http://bitbucket.org/eigen/eigen/get/3.3.2.tar.bz2
$ tar -xf 3.3.2.tar.bz2
$ mkdir -p external/include
$ mv eigen-eigen*/Eigen external/include/Eigen
```

An alternative is to look at the [Travis configuration file](./.travis.yml), which installs all the necessary dependencies
for use with continuous integration (automatically running the test suite in the cloud upon new commits).

Once you've got these dependencies in order, we will need to recursive initialize and update the git submodules:

```
$ git submodule update --init --recursive
```

This pulls down additional dependencies, and it might take some time.

Assuming all dependencies are correctly installed, we should hopefully be able to start compiling
the test binaries. This should be as simple as:

```
$ ./build.sh
```

This will build both debug- and release binaries for *crest* and its dependencies. This may take a long while.
Now is probably a good time to grab a cup of coffee and catch up on the latest news. If the build completes,
you should now have two new folders `target/debug/` and `target/release/` which correspond to debug- and
release builds, respectively. Let's see if we can run the test suite in release mode
(which is roughly a bazillion times quicker).

```
$ cd target/release
$ make check
```

If everything works, you should see a lot of reports of "OK" (successful) tests. If it doesn't,
you may perhaps try to run the debug build to see if it gives you any more information about what's wrong.


#### Running predetermined experiments

Since crest is a library and not an application, there is no canonical binary to run. However, in order to
study convergence etc., there is a test binary called `experiments`, which takes JSON input from standard input,
runs the chosen experiment and upon completion outputs error estimation, runtime and more. In order to understand
how it works, you will have to dig around in the source of `tests/experiment_runner.cpp`. As an example
of how to run it, go back to the **root** directory of the repository, and run the following:

```
$ cat schur_lumped_leapfrog.json
$ ./target/release/experiments < schur_lumped_leapfrog.json
```

This solves the wave equation on an L-shaped domain, using the Schur-based method to compute correctors
and the augmented mass-lumped Leapfrog method as an integrator. This might take a little time (took about 10 seconds on my system), before it spits out
a detailed JSON result with various information about the computation, including timing and
error estimates. Note that the vast majority of the time is spent in error computation
(reported as `transform_time` in the `timing` part of the online results in the output JSON).

There are also several Jupyter-based notebooks in `notebooks/` which makes it easy to run experiments for different
parameters and get an easy-to-read output, but I will not document these further.

## License

Copyright (c) 2016-2017 Andreas Borgen Longva

This project is licensed under the permissive MIT license. Please see the accompanying file `LICENSE` for legalese.
