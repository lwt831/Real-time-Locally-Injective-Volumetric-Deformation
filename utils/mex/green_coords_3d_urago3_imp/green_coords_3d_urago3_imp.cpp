#include <mex.h>
#include <string>
#include <vector>

#include"green_coords_3d_urago3.h"

const mxClassID mxPOINTER_CLASS = mxINDEX_CLASS;
using pointer_t = size_t;

//////////////////////////////////////////////////////////////////////////
/* memory management - redirect to MATLAB memory manager */
void* operator new(size_t size)
{
	void* ptr = mxMalloc(size);
	mexMakeMemoryPersistent(ptr);
	return ptr;
}
void* operator new[](size_t size)
{
	void* ptr = mxMalloc(size);
	mexMakeMemoryPersistent(ptr);
	return ptr;
}

void operator delete(void* ptr) { mxFree(ptr); }
void operator delete[](void* ptr) { mxFree(ptr); }


//////////////////////////////////////////////////////////////////////////

enum gc_stage {
	destroy_gc_computer = -1,
	create_gc_computer = 0, 
	compute_gc = 1,
	compute_gc_grad = 2,
	compute_gc_hess = 3,
};

#ifdef _DEBUG

#include "MatlabEngine.h"
#include <ctime>
#include<iostream>

int main()
{
	getMatEngine().connect("");
	const mxArray* x, * t;
	x = getMatEngine().getVariable("cx");
	t = getMatEngine().getVariable("ct");
	size_t nv = mxGetN(x);
	size_t nt = mxGetN(t);
	green_coords_3d_urago3<double>* gc_computer = new green_coords_3d_urago3<double>(mxGetDoubles(x), mxGetInt32s(t), nv, nt);

	const mxArray* p_in;
	p_in = getMatEngine().getVariable("testx");
	size_t np = mxGetN(p_in);


	std::clock_t start, end;
	double endtime;

	//////////////////////////////////////////////////
	/*test of compute phi & psi*/
	mxArray* phi, * psi;
	phi = mxCreateDoubleMatrix(np, gc_computer->get_nv(), mxREAL);
	psi = mxCreateDoubleMatrix(np, gc_computer->get_nt(), mxREAL);

	start = clock();
	gc_computer->compute_green_coords(mxGetDoubles(p_in), np, mxGetDoubles(phi), mxGetDoubles(psi));
	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "time of compute gc:" << endtime << std::endl;	
	getMatEngine().putVariable("phi_out", phi);
	getMatEngine().putVariable("psi_out", psi);

	//////////////////////////////////////////////////
	/*test of compute J_phi and J_psi*/
	mxArray* J_phi, * J_psi;
	J_phi = mxCreateDoubleMatrix(np * 3, gc_computer->get_nv(), mxREAL);
	J_psi = mxCreateDoubleMatrix(np * 3, gc_computer->get_nt(), mxREAL);

	start = clock();
	gc_computer->compute_green_coords_gradient(mxGetDoubles(p_in), np, mxGetDoubles(J_phi), mxGetDoubles(J_psi));
	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "time of compute J_gc:" << endtime << std::endl;
	getMatEngine().putVariable("J_phi_out", J_phi);
	getMatEngine().putVariable("J_psi_out", J_psi);

	//////////////////////////////////////////////////
	/*test of compute H_phi and H_psi*/
	mxArray* H_phi, * H_psi;
	H_phi = mxCreateDoubleMatrix(np * 5, gc_computer->get_nv(), mxREAL);
	H_psi = mxCreateDoubleMatrix(np * 5, gc_computer->get_nt(), mxREAL);

	start = clock();
	gc_computer->compute_green_coords_hessian(mxGetDoubles(p_in), np, mxGetDoubles(H_phi), mxGetDoubles(H_psi));
	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "time of compute H_gc:" << endtime << std::endl;
	getMatEngine().putVariable("H_phi_out", H_phi);
	getMatEngine().putVariable("H_psi_out", H_psi);
}

#else

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 1)
		mexErrMsgTxt("Invalid input: not enough input, green_coords_3d_urago3_imp(mode, ...)");
	const int mode = int(mxGetScalar(prhs[0]));

	if (mode == create_gc_computer)
	{
		if(nrhs < 3)
			mexErrMsgTxt("Invalid input: not enough input, handle = green_coords_3d_urago3_imp(mode, x, t);");
		const double* x = mxGetDoubles(prhs[1]);
		const int* t = mxGetInt32s(prhs[2]);
		size_t nv = mxGetN(prhs[1]);
		size_t nt = mxGetN(prhs[2]);

		green_coords_3d_urago3<double>* gc_computer = new green_coords_3d_urago3<double>(x, t, nv, nt);

		plhs[0] = mxCreateNumericMatrix(1, 1, mxPOINTER_CLASS, mxREAL);
		pointer_t* ptr_handle = (pointer_t*)mxGetData(plhs[0]);
		*ptr_handle = (pointer_t)gc_computer;
		return;
	}

	if (mode == destroy_gc_computer)
	{
		const pointer_t* ptr_handle = (pointer_t*)mxGetData(prhs[1]);
		green_coords_3d_urago3<double>* gc_computer = (green_coords_3d_urago3<double>*)(*ptr_handle);
		delete gc_computer;
		return;
	}

	if (mode > 0)
	{
		if(nrhs < 3)
			mexErrMsgTxt("Invalid input: not enough input, [phi, psi] = green_coords_3d_urago3_imp(mode, handle, p_in);");

		const pointer_t* ptr_handle = (pointer_t*)mxGetData(prhs[1]);
		green_coords_3d_urago3<double>* gc_computer = (green_coords_3d_urago3<double>*)(*ptr_handle);
		const double* p_in = mxGetDoubles(prhs[2]);
		int np = mxGetN(prhs[2]);	

		switch (mode)
		{
		case compute_gc:
		{
			plhs[0] = mxCreateDoubleMatrix(np, gc_computer->get_nv(), mxREAL);
			plhs[1] = mxCreateDoubleMatrix(np, gc_computer->get_nt(), mxREAL);
			double* phi = mxGetDoubles(plhs[0]);
			double* psi = mxGetDoubles(plhs[1]);
			gc_computer->compute_green_coords(p_in, np, phi, psi);
			return;
		}
		case compute_gc_grad:
		{
			plhs[0] = mxCreateDoubleMatrix(np * 3, gc_computer->get_nv(), mxREAL);
			plhs[1] = mxCreateDoubleMatrix(np * 3, gc_computer->get_nt(), mxREAL);
			double* J_phi = mxGetDoubles(plhs[0]);
			double* J_psi = mxGetDoubles(plhs[1]);
			gc_computer->compute_green_coords_gradient(p_in, np, J_phi, J_psi);
			return;
		}
		case compute_gc_hess:
		{
			plhs[0] = mxCreateDoubleMatrix(np * 5, gc_computer->get_nv(), mxREAL);
			plhs[1] = mxCreateDoubleMatrix(np * 5, gc_computer->get_nt(), mxREAL);
			double* H_phi = mxGetDoubles(plhs[0]);
			double* H_psi = mxGetDoubles(plhs[1]);
			gc_computer->compute_green_coords_hessian(p_in, np, H_phi, H_psi);
			return;
		}
		}
	}

}
#endif