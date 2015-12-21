# parralledl-lattice-boltzman

This is some  C code that implemented lattice boltzman method for solving Navier-Stokes equations using parallel computing system MPI, CUDA and OpenCL. Boundary conditions are set to be periodic boundary conditions in all four directions.

Lattice_Boltzmann2DMPI.c:
Input.txt specifies the dimension of the mesh, the dimensions of the 2d decompositions of the processors using MPI.
Sample 11 2 2 
would create a 11 by 11 mesh with 2 times 2 decomposition of processors, which is 4 processors in total for calling MPI

Lattice_BoltzmannCUDA.cu:
Implement lattice-boltzmann method  using CUDA on a gpu unit. Solves the equations on on a 2d mesh.

Lattice_BoltzmannOpenCL.c:

Implement lattice-boltzmann method  using OpenCL framework. Solves the equations on on a 2d mesh.
