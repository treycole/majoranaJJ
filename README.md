# Majoranas Modes in Periodic Array of Josephson Junctions
This project is for calculating the band structure for a periodic array of Josepshon Junctions that act like synthetic atoms. These repeating units are tested for topological phase structures in the presence of Zeeman field contributions, spin-orbit coupling, and particle-hole Bogoliubov de Gennes energies. We will test the known case for a Josephson Junction with a linear boundary between the 2DEG junction and the superconducting boundaries which is periodic in parallel to the boundary and show the expected topological phase transition at a superconducting phase difference of pi. We then test the scenario of a periodically structured superconducting region with nodular indentions making a "cross"-like junction. This unit is periodically repeated along the x-direction, being parallel to the junction. We also analyze the case of periodicity along the y-direction, perpendicular to the junction. 

## Repository Layout
The structure of this project has modules that can construct a variety of lattice shapes, operators to calculate the energy structure, and tests to visualize the wavefunction and energy band structure. The lattice functions are located in [majoranas/modules/lattice.py](majoranas/modules/lattice.py), the operator functions are located in [majoranas/modules/operators.py](majoranas/modules/operators.py), and the physical constants that are imported and used can be found in [majoranas/modules/constants.py](majoranas/modules/constants.py). In the [tests](/majoranas/tests) folder tests for a variety of physical systems can be found. Tests to visualize the wavefunction in both a periodic and non-periodic homogeneous unit cell for different scenarios are given to verify the validity of operator implementation. [Tests](/majoranas/tests) also includes the energy band structure for the superconducting and non-superconduction scenario of a homogeneous unit cell.

## Installation and Running Tests
To run the tests on your machine, clone the repository and add the repository directory to an environment variable named %PYTHONPATH%. The method of doing this varies depending on your operating system. On MacOS devices, open terminal and check the value of PYTHONPATH by executing: 

`$ echo $PYTHONPATH`

this will show you the directories that Python looks for when importing modules. To permanantly update your PYTHONPATH to include the directory of your project, you first need to open your `.bash_profile` in a text editor. You can do this by executing: 

`$ open TextEdit -a .bash_profile`

while in the directory that `.bash_profile` is stored, typically in the user profile folder. At the end of the file add the line:

`export PYTHONPATH = /path_to_directory`.

Now when you type `$ python test.py` on one of the tests in this repository, Python will know where to find the modules that the tests import. 


The dependencies are:
- Python 3
- Matplotlib
- NumPy
- Scipy


