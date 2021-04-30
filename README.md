This package contains the code that implements the following paper, 
"Real-time Locally Injective Volumetric Deformation"

What does the code contain.
The app is built with a combination of MATLAB, C++ code and mex/CUDA code. 
The C++ source code for the OpenGL UI with MS Visual Studio C++ project files is in the glvu folder.
The mex/CUDA source code for the GPU accelerated optimization is in the ProjHarmonicMap folder.
The mex/C++ for CPU accelerated optimization is in the utils folder.
Precompiled binary for UI and mex are provided with the package.

Requirements:
Windows 10
MATLAB(>=2019a)
A GLSL 3.3 compatible GPU
CUDA(Compute Capability>3.5)


To run the software:
1.Start MATLAB
2.cd to the code folder
3.call vol_Deformation_main.m. This will automatically open the GUI and load the "animal" model

The User Interface:
4.For deformation, the P2P constraint can be edited by:
	adding P2P constaints by left clicking on the model
	moving the P2P target by dragging and dropping any P2P constraint
	delete constraints by right clicking the P2P constraints

How to compile the binaries.
The following libraries are needed to compile the code
1.OpenGL GUI (glvu.exe)
	Eigen http://eigen.tuxfamily.org
	AntTweakBar http:///anttweakbar.sourceforge.net
	FreeGLUT http://freeglut.sourceforge.net
2.GPU mex file (ProjHarmonicMap.mexw64)
	CUDA toolkit(ver10.1 for precompiled mex) https://developer.nvidia.com/cuda-toolkit-archive
	cub https://nvlabs.github.io/cub/

