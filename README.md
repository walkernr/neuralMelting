

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

Once all of the required packages have been installed, simply cloning the directory and running the scripts is all that is needed. A shell script showing a convenient way to run the scripts is provided as `run.sh`.

Sample Results
==============

Here are some sample results for a small Lennard-Jones system of 108 atoms.

![Neural Network Classification](https://github.com/walkernr/neuralMelting/blob/master/images/neural_classification_probability.png)

![Neural Network Classified Feature Averages](https://github.com/walkernr/neuralMelting/blob/master/images/neural_classified_feature_average.png)

![Neural Network Melting Curve](https://github.com/walkernr/neuralMelting/blob/master/images/neural_melting_curve.png)

![Clustering Classification](https://github.com/walkernr/neuralMelting/blob/master/images/cluster_classification_probability.png)

![Clustering Classified Feature Averages](https://github.com/walkernr/neuralMelting/blob/master/images/cluster_classified_feature_average.png)

![t-SNE Sample Embedding with Clustering](https://github.com/walkernr/neuralMelting/blob/master/images/cluster_reduced_feature_embedding.png)

File Descriptions
=================

lammps_remcmc.py
----------------------------------

This program interfaces with LAMMPS to produce thermodynamic information and trajectories from NPT-HMC simulations that sweep through a range of temperatures and pressures simultaneously. LAMMPS is used to constuct the system, run the dynamics, and calculate the physical properties. The Monte Carlo moves, however, are performed in Python. Three types of Monte Carlo moves are defined and can be performed at each Monte Carlo sweep: The standard atom-wise position move (PMC), the NPT volume move (VMC), and the Hamiltonian Monte Carlo move (HMC). Different probabilities can be chosen for each type of MC move (specified for PMC and VMC while HMC takes the remaining probability). At the end of each data collection cycle, replica exchange Markov chain Monte Carlo is performed between all sample pairs.

The simulation can be run in multiple modes for debugging and implementing parallelism. There are boolean values for controlling verbosity, whether to run in parallel, if the parallelism is on a local or distributed cluster, and if the parallel method is multiprocessing or multithreading. The parallel implementation is done with the Dask Distributed library (or joblib for threading) and only runs the Monte Carlo simulations in parallel (the LAMMPS instances are serial). Since Dask uses Bokeh, multiprocessing runs may be monitored at localhost:8787/status assuming the default Bokeh TCP port is used.

General user-controlled parameters include the verbosity, parallel modes, number of workers/threads, simulation name, the element, the system size (in unit cells), the pressure and temperature ranges to be simulated, the cutoff for data collection, the number of samples to be collected, the frequency of data collection, the probability of the PMC and VMC moves, and the number of HMC timesteps. Material specific parameters are stored dictionaries with the atomic symbols serving as the keys ('LJ' for Lennard-Jones). The default is Lennard-Jones since it is useful for testing and is included with LAMMPS with no extra libraries needed. These dictionaries control the define the lattice type and parameter, the atomic mass, and the parameters for each MC move (position adjustment, box size adjustment, and timestep). The MC parameters are adaptively adjusted during the simulations for each sample independently.

Two output files are written to during the data collection cycles (assuming a low enough cutoff), one containing general thermodynamic properties and simulation details, and another containing the atom trajectories.

lammps_parse.py
---------------

This program parses the output from Monte Carlo simulations and pickles the data. The pickled data includes the temperatures, potential energies, kinetic energies, virial pressures, volumes, acceptation ratios for each MC move, and trajectories for each sample.

lammps_rdf.py
-------------------------

This program calculates the radial distributions, structure factors, and entropic fingerprints, and densities alongside the domains for each structural function for each sample using the pickled trajectory information from the parsing script. The calculations can be run in parallel using a multiprocessing or multithreading approach on local or distributed clusters. The parallelism is implemented with the Dask Distributed library. Since Dask uses Bokeh, multiprocessing runs may be monitored at localhost:8787/status assuming the default Bokeh TCP port is used. The rdf calculations are also optimized using a combination of vectorized code by way of NumPy alongside a JIT compiler by way of Numba.

lammps_neural.py
----------------

This program classifies samples as either solids or liquids by passing structural information through a multi-layer perceptron neural network.

### Structural Features
- Radial distribution
- Structure factor
- Entropic fingerprint

### Feature Scalers
- Standard: very common, vulnerable to outliers, does not guarantee a map to a common numerical range
- MinMax: also common, vulnerable to outliers, guarantees a map to a common numerical range
- Robust: resilient to outliers, does not guarantee a map to a common numerical range
- Tanh: resilient to outliers, guarantees a map to a common numerical range

### Feature Space Reducers
- None: use the raw scaled data
- PCA: common and fast, orthogonal linear transformation into new basis that maximizes variance of data along new projections
- Kernal PCA: slower than PCA, nonlinear reduction in the sample space rather than feature space
- Isomap: slower than PCA, a nonlinear reduction considered to be an extension of the Kernel PCA algorithm
- Locally Linear Embedding: slower than PCA, can be considered as a series of local PCA reductions that are globally stiched together

### Neural Networks
- 1-D Convolutional Neural Network Classifier

### Fitting Functions
- Logistic: well-behaved and easily extracted transition temperature estimate, symmetric

### Future Plans
- Refine neural network structures and hyperparameters with grid searching
- Add more neural networks
- Add more fitting functions

Since the structural features, feature scalers, feature space reducers, neural networks, and fitting functions are contained in libraries, the user may feel free to add their own.

lammps_cluster.py
-----------------

This program classifies samples as either solids or liquid by passing structural information through an unsupervised clustering algorithm following data scaling and feature space reduction into 2 dimensions. PCA reduction always performed with an option for further nonlinear feature space reduction.

### Structure Features
- Radial distribution
- Structure factor
- Entropic fingerprint

### Feature Scalers
- Standard: very common, vulnerable to outliers, does not guarantee a map to a common numerical range
- MinMax: also common, vulnerable to outliers, guarantees a map to a common numerical range
- Robust: resilient to outliers, does not guarantee a map to a common numerical range
- Tanh: resilient to outliers, guarantees a map to a common numerical range

### Feature Space Reducers
- None: use the PCA reduced data
- PCA: common and fast, orthogonal linear transformation into new basis that maximizes variance of data along new projections
- Kernal PCA: slower than PCA, nonlinear reduction in the sample space rather than feature space
- Isomap: slower than PCA, a nonlinear reduction considered to be an extension of the Kernel PCA algorithm
- Locally Linear Embedding: slower than PCA, can be considered as a series of local PCA reductions that are globally stiched together
- t-distributed Stochastic Neighbor Embedding: slowest method, treat affinities in original space as Gaussian distributed and transforms the data such that the new affinities are Student's t-distributed

### Clustering Methods
- K-Means: Good for globular data, struggles on elongated data sets and irregular cluster boundaries (including concentric clusters)
- Agglomerative: Good for globular data, struggles with low density clusters and concentric clusters
- Spectral: Good for connected data (including concentric), struggles with edges of globular data

### Fitting Functions
- Logistic: well-behaved and easily extracted transition temperature estimate, symmetric

### Future Plans
- Refine tuning of clustering method parameters
- Add more fitting functions

Since the structural features, feature scalers, feature space reducers, clustering methods, and fitting functions are contained in libraries, the user may feel free to add their own.

lammps_post.py
--------------

This program does post-processing on the transition predictions at multiple different pressures to extract the melting curve.

TanhScaler.py
-------------

This program implements a tanh-estimator as introduced by [Hampel et al](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118186435). with usage emulating the scalers found in the preprocessing library of Scikit-Learn. However, this implementation does not use the Hampel estimators and instead uses the means and standard deviations of the scores directly by way of the StandardScaler class from Scikit-Learn.

Further Development
===================

Some more advanced Monte Carlo methods may be implemented to improve sampling in addition to more robust machine learning analysis. GPU support for LAMMPS may also be added as well (if possible at all).