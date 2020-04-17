# Majoranas Modes in Josephson Junctions
This project is for calculating topological phases and testing for the presence of Majorana modes in a periodic array of Josepshon Junctions with varying geometries. The Junctions' band diagrams are tested for topological phase structures in the presence of Zeeman field contributions, spin-orbit coupling, and particle-hole coupling found from the Bogoliubov de Gennes Hamiltonian. We test the known case for a Josephson Junction with a linear boundary between the 2DEG junction and the superconducting leads. When this system is periodic in the direction parallel to the boundary, there is an expected topological phase transition at a superconducting phase difference of pi in the absence of a Zeeman field. 

## Repository Layout
The structure of this project has modules that can construct a variety of lattice shapes, operators to calculate the energy structure, and demos to visualize the wavefunction and energy band structure. 

The [etc folder](etc) contains the [physical constants](etc/constants.py) we use, plotting functions, and more stand alone modules that will be added in the future. 

The [lattice folder](lattice) contains the module for constructing lattices of a variety of [shapes](lattice/shapes.py) and the module for creating the [neighbor index arrays](lattice/neighbors.py). The neighbor index functions takes in the lattice shape array, which numbers the lattices sites 1 --> N acording to their position in the unit cell, and finds the index of the sites which are nearest neighbors to each lattice site. Thus they are N x 4 dimension matrices. There is a neighbor array for the bulk, and for the boundary to find the nearest neighbors in neighboring unit cells. Boundary arrays are only used when the system is periodic in the x or y direction. These neighbor arrays are used for calculating the Hamiltonians of the system. 

In the [operators folder](operators) there are two modules that use different methods to construct Hamiltonian matrices. The constructors in [sparsOP](operators/sparsOP.py) use Scipy's sparse matrix construction functions while the constructors in [densOP](operators/densOP.py) use NumPy's matrix construction operators. Scipy's sparse matrix functions save much of the memory and time required when diagonalizing large matrices whose elements are mostly zeros. They work equivalently when you only want the lower energy bands of the system. By using the neighbor and boundary arrays, descritized momentum operators can be constructed to create the Hamiltonian consisting of nearest neighbor interactions and spin-orbit coupling. For particle-hole symmetric systems, this can currently only be created a Josephson Junction system. This is a reflection of the way the Delta matrix in the BDG-Hamiltonian is defined. The optional parameters allow creation of a notch structure of varying sizes only the Josephson junction boundary. 

In the [demos folder](demos) tests for a variety of physical systems can be found, including the plots of their wavefunctions, band diagrams, and phase diagrams. These tests reveal the operation of the interconnected methods as well as give relevant physical results for the system we are studying. The demos are separated into sparse and dense folders, which tests both the sparse and dense matrix methods and shows that they work equivalently. In the [lattice sub-folder](demos/lattice) there are tests to plot the lattice and show that the modules to find the index of neighboring lattice sites are working correctly. Here is an image demonstrating the probability density of the ground state for a free electron inside a square unit cell. 

![](https://github.com/tbcole/majoranaJJ/blob/master/demos/images/sparse/wfs/sq_gs.png)

Here is an image demonstrating the probability density of the 39th excited state inside a donut shaped unit cell. 

![](https://github.com/tbcole/majoranaJJ/blob/master/demos/images/sparse/wfs/donut_fp39.png)

In the [comparisons folder](comparisons) contains tests to compare the time efficiency of new and old methods. Old methods are found in the [junk folder](junk).  

## Installation and Running Tests
To run the tests on your machine, clone the repository and add the repository directory to an environment variable named %PYTHONPATH%. The method of doing this varies depending on your operating system. On MacOS devices, open terminal and check the value of PYTHONPATH by executing: 

`$ echo $PYTHONPATH`

this will show you the directories that Python looks for when importing modules. To permanantly update your PYTHONPATH to include the directory of your project, you first need to open your `.bash_profile` in a text editor. You can do this by executing: 

`$ open -a TextEdit .bash_profile`

while in the directory that `.bash_profile` is stored, typically in the user profile folder. At the end of the file add the line:

`export PYTHONPATH = /path_to_project`.

Now when you type `$ python test.py` on one of the tests in this repository, Python will know where to find the modules that the tests import. 


The dependencies are:
- Python 3
- Matplotlib
- NumPy
- Scipy


