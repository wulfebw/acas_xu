# Deep_Collision_Avoidance

This repository contains algorithms to implement and visualize deep learning for UAS collision avoidance.

DeepRL_5.py contains the python code to run the Deep RL algorithm. Important considerations include:
* Values of batch_size, train_freq, k_size, and lamda values. Tuning these parameters changes the performance of the algorithm
* Reward Shape: The reward is composed of three penalties, with the first being the most important. The shape of this reward was recently changed from a smoothed step function to a triangle shape, so that the ownship is penalized more for getting closer to the intruder even in close range
* Sampling of experiences: Prioritized experience replay could be very helpful. Right now experiences are drawn randomly

Plot_DRL.ipynb is very similar to DeepRL_5.py but is tailored to plotting a picture of the encounter rather than performing deep reinforcment learning. This was useful when trying to debug the dynamics.

The remaining files in DeepRL work to plot the resulting policy produced by the DeepRL algorithm. The algorithm automatically produces .h5 files at specified intervals. These files are converted to .nnet text files using convertHDF5.py. These .nnet files are then passed to a C++ shared library, which loads the neural network weights and quickly computes the Q values given a set or multiple sets of inputs. This C++ code is found in the nnet_blas.cpp and nnet_blas.hpp files. These files are compiled to a shared library and called from within Julia using ccall.

The policy plots produced are shown in the AA229 paper. Note that these plots look very similar to the plots found in short-term-conf-reso.pdf, but with some differences, as discussed in the paper. We need to figure out if the DeepRL algorithm is making mistakes, or if parts of the problem like the rewards and actions were changed enough to generate different but valid policies. Eventually, this algorithm will be scaled up to a more complex 3D realm, so we want to make sure that the algorithm is correct before moving on.
