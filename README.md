

     ███▄    █ ▓█████  █    ██  ██▀███   ▄▄▄       ██▓        ███▄ ▄███▓▓█████  ██▓  ▄▄▄█████▓ ██▓ ███▄    █   ▄████
     ██ ▀█   █ ▓█   ▀  ██  ▓██▒▓██ ▒ ██▒▒████▄    ▓██▒       ▓██▒▀█▀ ██▒▓█   ▀ ▓██▒  ▓  ██▒ ▓▒▓██▒ ██ ▀█   █  ██▒ ▀█▒
    ▓██  ▀█ ██▒▒███   ▓██  ▒██░▓██ ░▄█ ▒▒██  ▀█▄  ▒██░       ▓██    ▓██░▒███   ▒██░  ▒ ▓██░ ▒░▒██▒▓██  ▀█ ██▒▒██░▄▄▄░
    ▓██▒  ▐▌██▒▒▓█  ▄ ▓▓█  ░██░▒██▀▀█▄  ░██▄▄▄▄██ ▒██░       ▒██    ▒██ ▒▓█  ▄ ▒██░  ░ ▓██▓ ░ ░██░▓██▒  ▐▌██▒░▓█  ██▓
    ▒██░   ▓██░░▒████▒▒▒█████▓ ░██▓ ▒██▒ ▓█   ▓██▒░██████▒   ▒██▒   ░██▒░▒████▒░██████▒▒██▒ ░ ░██░▒██░   ▓██░░▒▓███▀▒
    ░ ▒░   ▒ ▒ ░░ ▒░ ░░▒▓▒ ▒ ▒ ░ ▒▓ ░▒▓░ ▒▒   ▓▒█░░ ▒░▓  ░   ░ ▒░   ░  ░░░ ▒░ ░░ ▒░▓  ░▒ ░░   ░▓  ░ ▒░   ▒ ▒  ░▒   ▒
    ░ ░░   ░ ▒░ ░ ░  ░░░▒░ ░ ░   ░▒ ░ ▒░  ▒   ▒▒ ░░ ░ ▒  ░   ░  ░      ░ ░ ░  ░░ ░ ▒  ░  ░     ▒ ░░ ░░   ░ ▒░  ░   ░
       ░   ░ ░    ░    ░░░ ░ ░   ░░   ░   ░   ▒     ░ ░      ░      ░      ░     ░ ░   ░       ▒ ░   ░   ░ ░ ░ ░   ░
             ░    ░  ░   ░        ░           ░  ░    ░  ░          ░      ░  ░    ░  ░        ░           ░       ░


Description
===========

This project contains a collection of scripts for estimating the melting point of a material using an empirical machine learning approach.

Assuming that a transition in a physical system is accompanied by a change in the structure of the system, a classification approach can be utilized to predict the location of the transition by partitioning the physical data into two phases. The physical data used in this case is configurations of atoms generated with a Hamiltonian Monte Carlo algorithm. Replica exchange Markov chain Monte Carlo is also implemented to apply parallel tempering and attempt to over come undercooling/overheating effects. Ample control of simulation parameters is provided via command line flags.

Structural information from the Monte Carlo samples are used as the input features for both supervised and unsupervised learning. Many different data preparation and analysis techniques are provided including a selection of available features to train on, feature scaling, linear/nonlinear feature space reductions, clustering methods, and more to come. Melting point predictions are done with sigmoid fits to the plot of liquid phase probability as a function of temperature.

Requirements
============

Python
------

I recommend using [Anaconda](https://anaconda.org/).

- [LAMMPS](https://lammps.sandia.gov/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Dask](https://dask.org/)
- [joblib](https://pypi.org/project/joblib/)
- [Numba](http://numba.pydata.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)/[Theano](http://deeplearning.net/software/theano/)
- [tqdm](https://pypi.org/project/tqdm/)
- [MatPlotLib](https://matplotlib.org/)

The scripts are written for Python 3.X syntax, but minor adjustments allow them to comply with Python 2.X syntax.

LAMMPS Information
------------------

In order to use LAMMPS in Python, it must be compiled as a shared library. The instructions for doing so are presented on the LAMMPS webpage. Alternatively, it can be installed from the `conda-forge` Anaconda channel.

Installation and Execution
==========================

Once all of the required packages have been installed, simply cloning the directory and running the scripts is all that is needed. Use `--help` or `-h` for help. A shell script showing a convenient way to run the scripts is provided as `run.sh`.

File Descriptions
=================

lammps_remcmc.py
----------------------------------

This program interfaces with LAMMPS to produce thermodynamic information and trajectories from NPT-HMC simulations that sweep through a range of temperatures and pressures simultaneously. LAMMPS is used to constuct the system, run the dynamics, and calculate the physical properties. The Monte Carlo moves, however, are performed in Python. Three types of Monte Carlo moves are defined and can be performed at each Monte Carlo sweep: The standard atom-wise position move (PMC), the NPT volume move (VMC), and the Hamiltonian Monte Carlo move (HMC). Different probabilities can be chosen for each type of MC move (specified for PMC and VMC while HMC takes the remaining probability). At the end of each data collection cycle, replica exchange Markov chain Monte Carlo is performed between all sample pairs.

The simulation can be run in multiple modes for debugging and implementing parallelism. There are boolean values for controlling verbosity, whether to run in parallel, if the parallelism is on a local or distributed cluster, and if the parallel method is multiprocessing or multithreading. The parallel implementation is done with the Dask Distributed library (or joblib for threading) and only runs the Monte Carlo simulations in parallel (the LAMMPS instances are serial). Since Dask uses Bokeh, multiprocessing runs may be monitored at `localhost:8787/status` assuming the default Bokeh TCP port is used.

General user-controlled parameters include the verbosity, parallel modes, number of workers/threads, simulation name, the element, the system size (in unit cells), the pressure and temperature ranges to be simulated, the cutoff for data collection, the number of samples to be collected, the frequency of data collection, the probability of the PMC and VMC moves, and the number of HMC timesteps. Material specific parameters are stored dictionaries with the atomic symbols serving as the keys ('LJ' for Lennard-Jones). The default is Lennard-Jones since it is useful for testing and is included with LAMMPS with no extra libraries needed. These dictionaries control the define the lattice type and parameter, the atomic mass, and the parameters for each MC move (position adjustment, box size adjustment, and timestep). The MC parameters are adaptively adjusted during the simulations for each sample independently.

Two output files are written to during the data collection cycles (assuming a low enough cutoff), one containing general thermodynamic properties and simulation details, and another containing the atom trajectories.

lammps_parse.py
---------------

This program parses the output from Monte Carlo simulations and dumps the data. The dumped data includes the temperatures, potential energies, kinetic energies, virial pressures, volumes, acceptation ratios for each MC move, and trajectories for each sample.

lammps_distr.py
-------------------------

This program calculates the radial distributions, structure factors, and entropic fingerprints, densities, and cartesian volume densities alongside the domains for each structural function for each sample using the dumped trajectory information from the parsing script. The calculations can be run in parallel using a multiprocessing or multithreading approach on local or distributed clusters. The parallelism is implemented with the Dask Distributed library. Since Dask uses Bokeh, multiprocessing runs may be monitored at localhost:8787/status assuming the default Bokeh TCP port is used. The rdf calculations are also optimized using a combination of vectorized code by way of NumPy alongside a JIT compiler by way of Numba.

lammps_vae.py
----------------

This program encodes the sample structure information with a variational autoencoder implemented with a tunable latent dimension and 3-dimensional convolution layers in the encoder and decoder networks. Further unsupervised analysis is performed on the latent encodings.

### Structural Features
- Cartesian volume densities

### Feature Scalers
- Standard: very common, vulnerable to outliers, does not guarantee a map to a common numerical range
- MinMax: also common, vulnerable to outliers, guarantees a map to a common numerical range
- Robust: resilient to outliers, does not guarantee a map to a common numerical range
- Tanh: resilient to outliers, guarantees a map to a common numerical range

### Neural Networks
- 3-dimensional convolutional variational autoencoder with a Gaussian prior (KL-divergence used as regularizer)

### Optimizers
- SGD: standard stochastic gradient descent
- Adadelta: an optimizer (extension of Adagrad) with parameter-specific adaptive learning rates based on past gradient updates
- Adam: an optimizer (combination of Adagrad and RMSprop) with parameter-specific adaptive learning rates based on parameter update frequency and rate of change
- Nadam: and optmizer that implements Nesterov momentum in the Adam optmizer

### Loss Functions
- MSE: mean squared error difference between target and prediction
- Binary Crossentropy: cross entropy between target and prediction (ranges from 0 to 1)

### Manifold Learning
- PCA: common and fast, orthogonal linear transformation into new basis that maximizes variance of data along new projections
- Kernal PCA: slower than PCA, nonlinear reduction in the sample space rather than feature space
- Isomap: slower than PCA, a nonlinear reduction considered to be an extension of the Kernel PCA algorithm
- Locally Linear Embedding: slower than PCA, can be considered as a series of local PCA reductions that are globally stiched together
- t-distributed Stochastic Neighbor Embedding: slowest method, treat affinities in original space as Gaussian distributed and transforms the data such that the new affinities are Student's t-distributed in a lower dimensional space with identical affinities to the original space

### Fitting Functions
- Logistic: well-behaved and easily extracted transition temperature estimate, symmetric

### Future Plans
- Add support for more structural functions
- Refine neural network structures and hyperparameters with grid searching
- Add more neural network structures

Dictionaries are used for many of the options, so the user is free to add more

TanhScaler.py
-------------

This program implements a tanh-estimator as introduced by [Hampel et al](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118186435). with usage emulating the scalers found in the preprocessing library of Scikit-Learn. However, this implementation does not use the Hampel estimators and instead uses the means and standard deviations of the scores directly by way of the StandardScaler class from Scikit-Learn.

Further Development
===================

Some more advanced Monte Carlo methods may be implemented to improve sampling in addition to more robust machine learning analysis. GPU support for LAMMPS may also be added as well (if possible at all).