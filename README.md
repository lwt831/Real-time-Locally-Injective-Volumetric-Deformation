This package contains the code that implements the following paper, 
"Real-time Locally Injective Volumetric Deformation"

What does the code contain.
==============
The app is built with a combination of MATLAB, C++ code and mex/CUDA code. <br>
The C++ source code for the OpenGL UI with MS Visual Studio C++ project files is in the glvu folder.<br>
The mex/CUDA source code for the GPU accelerated optimization is in the ProjHarmonicMap folder.<br>
The mex/C++ for CPU accelerated optimization is in the utils folder.<br>
Precompiled binary for UI and mex are provided with the package.<br>

Requirements:
==============
* Windows 10<br>
* MATLAB(>=2019a)<br>
* A GLSL 3.3 compatible GPU<br>
* CUDA(Compute Capability>3.5)<br>


To run the software:
==============
1.Start MATLAB<br>
2.cd to the code folder<br>
3.call vol_Deformation_main.m. This will automatically open the GUI and load the "animal" model<br>

The User Interface:
--------------
4.For deformation, the P2P constraint can be edited by:<br>
>adding P2P constaints by left clicking on the model<br>
>moving the P2P target by dragging and dropping any P2P constraint<br>
>delete constraints by right clicking the P2P constraints<br>

How to compile the binaries.
==============
The following libraries are needed to compile the code<br>
1.OpenGL GUI (glvu.exe)<br>
* Eigen http://eigen.tuxfamily.org<br>
* AntTweakBar http://anttweakbar.sourceforge.net<br>
* FreeGLUT http://freeglut.sourceforge.net<br>

2.GPU mex file (ProjHarmonicMap.mexw64)<br>
* CUDA toolkit(ver10.1 for precompiled mex) https://developer.nvidia.com/cuda-toolkit-archive<br>
* cub https://nvlabs.github.io/cub/<br>

