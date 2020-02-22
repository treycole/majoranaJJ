# Majoranas in Synthetic Atoms
This project is for calculating the band structure for a periodic array of synthetic atoms. These synthetic atoms are modeled by a unit cell that can be repeated periodically. The structure of this project has modules that can construct a variety of lattice shapes, operators to calculate the energy structure, and tests to visualize the wavefunction and energy band structure.

The lattice functions are located in [/Modules/lattice.py](/Modules/lattice.py).
The operator functions are located in [/Modules/operators.py](/Modules/operators.py).
The physical constants that are imported and used can be found in [/Modules/constants.py](/Modules/constants.py).

In the [tests](Tests) folder tests for a variety of physical systems can be found. Tests to visualize the wavefunction in a unit cell for a [free particle](Tests/Free%20Particle), with the addition of [spin-orbit coupling](Tests/Spin-Orbit-Coupling) and [superconductivity](Tests/Superconductivity), are given to verify the validity of operator implementation. Tests include the implementation of periodicty into the energy structure in order to [visualize the band structure](Tests/Periodic) are included.
