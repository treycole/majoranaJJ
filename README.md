# Majoranas Modes in Periodic Array of Josephson Junctions
This project is for calculating the band structure for a periodic array of Josepshon Junctions that act like synthetic atoms. These repeating units are tested for topological phase structures in the presence of Zeeman field contributions, spin-orbit coupling, and particle-hole Bogoliubov de Gennes energies. We will test the known case for a Josephson Junction with a linear boundary between the 2DEG junction and the superconducting boundaries which is periodic in parallel to the boundary and show the expected topological phase transition at a superconducting phase difference of pi. We then test the scenario of a periodically structured superconducting region with nodular indentions making a "cross"-like junction. This unit is periodically repeated along the x-direction, being parallel to the junction. We also analyze the case of periodicity along the y-direction, perpendicular to the junction. 

The structure of this project has modules that can construct a variety of lattice shapes, operators to calculate the energy structure, and tests to visualize the wavefunction and energy band structure. The lattice functions are located in [majoranas/modules/lattice.py](majoranas/modules/lattice.py), the operator functions are located in [majoranas/modules/operators.py](majoranas/modules/operators.py), and the physical constants that are imported and used can be found in [majoranas/modules/constants.py](majoranas/modules/constants.py).

In the [tests](/majoranas/tests) folder tests for a variety of physical systems can be found. Tests to visualize the wavefunction in both a periodic and non-periodic homogeneous unit cell for different scenarios are given to verify the validity of operator implementation. [Tests](/majoranas/tests) also includes the energy band structure for the superconducting and non-superconduction scenario of a homogeneous unit cell.

To run the tests on your machine, clone the repository and add the repository directory to an environment variable named %PYTHONPATH%. The method of doing this varies depending on your operating system and can be readily found via an internet search. The current dependencies are:
- Python interpreter
- Matplotlib
- NumPy
- Scipy


