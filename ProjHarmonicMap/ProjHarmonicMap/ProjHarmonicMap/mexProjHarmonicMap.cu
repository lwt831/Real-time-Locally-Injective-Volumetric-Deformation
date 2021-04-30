#include <stdio.h>
#ifdef _DEBUG
#include "MatlabEngine.h"
#endif
#include <mex.h>
#include "harmonicmap.cuh"


static vec newCageCoef;

template<typename R>
R getFieldValueWithDefault(const mxArray* mat, const char* name, R defaultvalue)
{
	const mxArray* f = mat ? mxGetField(mat, 0, name) : nullptr;
	return f ? R(mxGetScalar(f)) : defaultvalue;
}

void MallocForModel()
{
	cageX_in.resize(3 * nv);
	cageT_in.resize(3 * nf);

	x1.resize(nf * 3);
	x2.resize(nf * 3);
	x3.resize(nf * 3);
	N.resize(nf * 3);
	nn.resize(nf);
	dx1.resize(nf * 3);
	dx2.resize(nf * 3);
	dx3.resize(nf * 3);
	l1a.resize(nf);
	l2a.resize(nf);
	l3a.resize(nf);

	cNdx1.resize(nf * 3);
	cNdx2.resize(nf * 3);
	cNdx3.resize(nf * 3);

	ProjMat.resize(nvnf, nvar);

	D.resize(nd, nvar);

	Half_edge_Info.resize(ne * 4);
	Kappa.resize(3 * ne, nvar);

	smooth_energy_h.resize(nvar, nvar);
	viewY.resize(nd * 3);
}




template<class R>
void MallocForHMDeformer(std::vector<real> cpu_para)
{

	Jp.resize(nvar, 3 * nj);
	JpSp.resize(nvar, 3 * nj);
	J.resize(3 * nj, 3);
	U.resize(3, 3 * nj);
	V.resize(3, 3 * nj);
	S.resize(3 * nj);


	delta.resize(nvar * 3);
	energy_h.resize(nvar * 3, nvar * 3);
	eh_temp.resize(3 * nvar * 3 * nvar);
	M.resize(nvar, nvar);
	b.resize(nvar * 3);

	/*choose chol or lu solver*/
	if (solver_type == 0)
	{
		energy_h.LUSolver_init();
	}
	else if (solver_type == 1)
	{
		energy_h.CholSolver_init();
	}


	energy_g.resize(nvar * 3);

	para = cpu_para;

	dev_sample_radius = para.pdata;
	dev_w_p2p = dev_sample_radius + 1;
	dev_w_smooth = dev_w_p2p + 1;
	dev_h_rate = dev_w_smooth + 1;
	lips_constant = dev_h_rate + 1;
	dev_ls_step_size = lips_constant + 1;


	scalars4linesearch.resize(4);
	dot_delta_g = scalars4linesearch.pdata;	//1 scalar
	q_coef = dot_delta_g + 1;	//3 scalars
	dInfo.resize(8 * 5);

	dis_energy_gradient_resp2J.resize(9 * nj);
	Q.resize(81 * nh);

	temp.resize(9 * nh, 3 * nvar);
	C.resize(27 * nh * nvar);

	z.resize(nvar * 3);
	eq_g.resize(nvar * 3);


	DevArrayQ.resize(nh * 3);
	DevArrayJp.resize(nh * 3);
	DevArrayC.resize(nh * 3);

	ka_norm.resize(ne * 2);

	isp.resize(nj * 3);

	adaptive_sampling_position.resize(max_number_adaptive_sampling * 6);
	flip_point_idx.resize(max_number_adaptive_sampling * 2);
	flag_list_of_failed.resize(max_number_adaptive_sampling);
	d_temp_storage.resize(max_number_adaptive_sampling);
	max_idx_array.resize(max_number_adaptive_sampling);
	J_tmp_storage.resize(max_number_adaptive_sampling * 9);
	H_norm_storage.resize(max_number_adaptive_sampling);
	adaptive_sampling_father_knot_idx.resize(max_number_adaptive_sampling * 2);

	num_flip_points.resize(1);
	tmp_s3.resize(1);
	bd.resize(4);
	newCageCoef.resize(3 * nvnf);

}

void createHdl()
{
	cublasCreate(&hdl);
	cublasSetPointerMode(hdl, CUBLAS_POINTER_MODE_DEVICE);

	const std::vector<real> cpu_constants = { 1, 0, -1, 2, -2, 0.5 };
	constants = cpu_constants;
	dev_one = constants.pdata;
	dev_zero = dev_one + 1;
	dev_minus_one = dev_zero + 1;
	dev_two = dev_minus_one + 1;
	dev_minus_two = dev_two + 1;
	dev_half = dev_minus_two + 1;

	for (int i = 0; i < 9; i++)
		cudaEventCreate(&times[i]);

}

void cleanup() {
	printf("MEX-file is terminating, destroying array\n");
	if (hdl)
	{
		cublasDestroy(hdl);
	}

	for (int i = 0; i < 9; i++)
	{
		cudaEventDestroy(times[i]);
	}
	/*
	cudaDeviceReset();*/
	printf("Freed CUDA Array.\n");
}

template<class R>
int eigenvalueDecomposion(cuMatrix<R> &A, cuVector<R> &d)
{
	if (A.dims[0] != A.dims[1])
	{
		printf("Matrix must be symmetric!\n");
		return -1;
	}
	int m = A.dims[0];
	if (d.len != A.dims[0])
	{
		d.resize(m);
	}
	cusolverDnHandle_t cusolverH;
	cuVector<R> d_work;
	cuVector<int> info;
	info.resize(1);
	int lwork;
	cusolverDnCreate(&cusolverH);
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, A.pdata, m, d.pdata, &lwork);
	d_work.resize(lwork);
	cusolverStatus_t cusolver_status = cusolverDnDsyevd(cusolverH, jobz, uplo, m, A.pdata, m, d.pdata, d_work.pdata, lwork, info.pdata);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR;

	return 0;
}

#ifdef _DEBUG
void putGPUData2Matlab(real* pdata, int m, int n, std::string name)
{
	mxArray* mx;
	mx = mxCreateDoubleMatrix(m, n, mxREAL);
	cudaMemcpy(mxGetDoubles(mx), pdata, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	getMatEngine().putVariable(name, mx);
}
#endif


void set_model(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nvar == 0) mexErrMsgTxt("Please set number of DOFs at First!\n");
	if (nrhs != 6) mexErrMsgTxt("origin_var = ProjHarmonicMap('set_model', cageX, cageT, edgeInfo, x, proj);\n");
	const mxArray* cageX = (prhs[1]);
	const mxArray* cageT = (prhs[2]);
	const mxArray* edgeInfo = (prhs[3]);
	const mxArray* x = (prhs[4]);

	nv = mxGetN(cageX);
	nf = mxGetN(cageT);
	ne = nf + nv - 2;
	nvnf = nv + nf;
	if (nvar > nvnf)
	{
		nvar = nvnf;
		printf("DOFs must be less than number of basis functions of VHM\n");
		printf("reset DOFs = %d\n", nvar);
	}
	nd = mxGetN(x);

	mexPrintf("model initializing, GPU memorry allocation\n");
	MallocForModel();
	mexPrintf("success!\n");

	vec x_in(nd * 3);

	cudaMemcpyAsync(cageX_in.pdata, mxGetDoubles(cageX), 3 * nv * sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(cageT_in.pdata, mxGetInt32s(cageT), 3 * nf * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(Half_edge_Info.pdata, mxGetInt32s(edgeInfo), sizeof(int) * 4 * ne, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(x_in.pdata, mxGetDoubles(x), 3 * nd * sizeof(real), cudaMemcpyHostToDevice);

	precompute_cageGeometry << <blockNum(nf), threadsPerBlock >> > (cageX_in.pdata, cageT_in.pdata, nv, nf, x1.pdata, x2.pdata, x3.pdata, N.pdata, nn.pdata, dx1.pdata, dx2.pdata, dx3.pdata, l1a.pdata, l2a.pdata, l3a.pdata, cNdx1.pdata, cNdx2.pdata, cNdx3.pdata);

	mat temp_Kappa;
	temp_Kappa.resize(3 * ne, nvnf);
	temp_Kappa.zero_fill();
	Lipschitz_Kernel_Matrix_Kappa << <blockNum(ne), threadsPerBlock >> > (cageX_in.pdata, nv, cageT_in.pdata, Half_edge_Info.pdata, ne, N.pdata, nn.pdata, temp_Kappa.pdata);
	/*
	mat temp_smooth_energy_h;
	temp_smooth_energy_h.resize(nvnf, nvnf);
	syrk(hdl, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, nvnf, 3 * ne, dev_one, temp_Kappa.pdata, 3 * ne, dev_zero, temp_smooth_energy_h.pdata, nvnf);
	

	vec temp_d;
	temp_d.resize(nvnf);
	eigenvalueDecomposion(temp_smooth_energy_h, temp_d);

#ifdef _DEBUG
	mxArray* mxProj;
	mxProj = mxCreateDoubleMatrix(nvnf, nvar, mxREAL);
	mxProj = getMatEngine().getVariable("proj");
	cudaMemcpy(temp_smooth_energy_h.pdata, mxGetDoubles(mxProj), nvnf * nvar * sizeof(real), cudaMemcpyHostToDevice);
#endif

	cudaMemcpyAsync(ProjMat.pdata, temp_smooth_energy_h.pdata, nvnf * nvar * sizeof(real), cudaMemcpyDeviceToDevice);
	*/
	cudaMemcpyAsync(ProjMat.pdata, mxGetDoubles(prhs[5]), nvnf * nvar * sizeof(real), cudaMemcpyHostToDevice);
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, 3 * ne, nvar, nvnf, dev_one, temp_Kappa.pdata, 3 * ne, ProjMat.pdata, nvnf, dev_zero, Kappa.pdata, 3 * ne);
	gemm(hdl, CUBLAS_OP_T, CUBLAS_OP_N, nvar, nvar, 3 * ne, dev_one, Kappa.pdata, 3 * ne, Kappa.pdata, 3 * ne, dev_zero, smooth_energy_h.pdata, nvar); // maybe syrk is ok.

	mat temp_D;
	temp_D.resize(nd, nvnf);
	real* phi_out = temp_D.pdata;
	real* psi_out = phi_out + nd * nv;
	temp_D.zero_fill();

	green_coords_3d_urago3_vectorized2 << <nd, threadsPerBlock >> > (cageT_in.pdata, nf, x1.pdata, x2.pdata, x3.pdata, N.pdata, nn.pdata, dx1.pdata, dx2.pdata, dx3.pdata, l1a.pdata, l2a.pdata, l3a.pdata, x_in.pdata,
		nd, phi_out, psi_out);
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nd, nvar, nvnf, dev_one, temp_D.pdata, nd, ProjMat.pdata, nvnf, dev_zero, D.pdata, nd);

	vec vhm_coef;
	vhm_coef.resize(nvnf * 3);
	cudaMemcpyAsync(vhm_coef.pdata, cageX_in.pdata, nv * 3 * sizeof(real), cudaMemcpyDeviceToDevice);
	cudaMemcpyAsync(vhm_coef.pdata + nv * 3, N.pdata, nf * 3 * sizeof(real), cudaMemcpyDeviceToDevice);
	vec Proj_coef;
	Proj_coef.resize(nvar * 3);
	gemm(hdl, CUBLAS_OP_T, CUBLAS_OP_T, nvar, 3, nvnf, dev_one, ProjMat.pdata, nvnf, vhm_coef.pdata, 3, dev_zero, Proj_coef.pdata, nvar);

	mxArray* out1;
	
	plhs[0] = mxCreateDoubleMatrix(nvar, 3, mxREAL);
	cudaMemcpyAsync(mxGetDoubles(plhs[0]), Proj_coef.pdata, nvar * 3 * sizeof(real), cudaMemcpyDeviceToHost);

	Proj_delta.resize(nvnf * 3);
	Proj_z.resize(nvnf * 3);
}

void set_deformer(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs != 3) mexErrMsgTxt("ProjHarmonicMap('set_Deformer', isometric_sample_points, ipara);");
	const mxArray* isometric_sample_points = prhs[1];
	const mxArray* ipara = prhs[2];



	nj = mxGetN(isometric_sample_points);

	ss = getFieldValueWithDefault<int>(ipara, "hess_sample_step_size", 100);
	//et = getFieldValueWithDefault<int>(ipara, "energy_type", 1);
	et = 1;

	nh = floor(real(nj - 1) / ss) + 1;

	real epsilon = getFieldValueWithDefault<real>(ipara, "epsilon", 0); // sampling radius of interior
	real w_p2p = getFieldValueWithDefault<real>(ipara, "w_p2p", 100);
	real w_smooth = getFieldValueWithDefault<real>(ipara, "w_smooth", 1);
	solver_type = getFieldValueWithDefault<int>(ipara, "solver", 0); // 0 for Lu, 1 for cholesky
	showTimeDetails = getFieldValueWithDefault<int>(ipara, "showTimeDetails", 1);
	real h_rate = real(1) / nh;
	real ls_step_size = 1.0;
	std::vector<real> cpu_para = { epsilon, w_p2p, w_smooth, h_rate ,0.0, ls_step_size };

	mexPrintf("HMDeformer initializing, GPU memorry allocation\n");
	MallocForHMDeformer<real>(cpu_para);
	mexPrintf("success!\n");
	


	generate_max_index_array << <blockNum(max_number_adaptive_sampling), threadsPerBlock >> > (max_number_adaptive_sampling, max_idx_array.pdata);


	cudaMemcpyAsync(isp.pdata, mxGetDoubles(isometric_sample_points), 3 * nj * sizeof(real), cudaMemcpyHostToDevice);


	mat temp_Jpt(3 * nj, nvnf);
	real* Jphi = temp_Jpt.pdata;
	real* Jpsi = Jphi + 3 * nj * nv;
	green_coords_3d_urago3_gradient_vectorized2 << <nj, threadsPerBlock >> > (cageT_in.pdata, nf, x1.pdata, x2.pdata, x3.pdata, N.pdata, nn.pdata, dx1.pdata, dx2.pdata, dx3.pdata, l1a.pdata, l2a.pdata, l3a.pdata,
		isp.pdata, nj, Jphi, Jpsi); //compute gradient sampling points of interior

	gemm(hdl, CUBLAS_OP_T, CUBLAS_OP_T, nvar, 3 * nj, nvnf, dev_one, ProjMat.pdata, nvnf, temp_Jpt.pdata, 3 * nj, dev_zero, Jp.pdata, nvar);
	convertType << <blockNum(nvar * 3 * nj), threadsPerBlock >> > (JpSp.pdata, Jp.pdata, nvar * 3 * nj);
	float* C1 = temp.pdata;
	float* C2 = temp.pdata + 9 * nh * nvar;
	float* C3 = temp.pdata + 18 * nh * nvar;
	std::vector<const float*> HostArrayQ, HostArrayJp;
	std::vector<float*> HostArrayC;
	HostArrayQ.resize(3 * nh);
	HostArrayJp.resize(3 * nh);
	HostArrayC.resize(3 * nh);
	for (int i = 0; i < nh; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			HostArrayQ[3 * i + j] = Q.pdata + (3 * i + j) * 27;
			HostArrayJp[3 * i + j] = JpSp.pdata + i * ss * 3 * nvar;
			//HostArrayC[3 * i + j] = C.pdata + 9 * nvar * nh * j + i * 9 * nvar;
		}
		HostArrayC[3 * i] = C1 + i * 9 * nvar;
		HostArrayC[3 * i + 1] = C2 + i * 9 * nvar;
		HostArrayC[3 * i + 2] = C3 + i * 9 * nvar;
	}

	DevArrayQ = HostArrayQ;
	DevArrayJp = HostArrayJp;
	DevArrayC = HostArrayC;
	dev_const_S = std::vector<float>{ 0.0f, 1.0f, float(h_rate) };
}

void set_solver(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs != 2) mexErrMsgTxt("ProjHarmonicMap('set_solver', ipara);");
	const mxArray* ipara = prhs[1];
	solver_type = getFieldValueWithDefault<int>(ipara, "solver", 0); // 0 for Lu, 1 for cholesky
	if (solver_type == 0)
	{
		energy_h.LUSolver_init();
		printf("solver = LU solver!\n");
	}
	else if (solver_type == 1)
	{
		energy_h.CholSolver_init();
		printf("solver = Cholesky solver!\n");
	}
}


void set_ls_step_size(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs > 2) mexErrMsgTxt("ProjHarmonicMap('set_ls_step_size', ls_step_size;");
	cudaMemcpyAsync(dev_ls_step_size, mxGetDoubles(prhs[1]), sizeof(real), cudaMemcpyHostToDevice);

}

void set_sampling_points(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs > 3) mexErrMsgTxt("ProjHarmonicMap('set_sampling_points', lips_sampling_points, eps);");
	if (!sample_radius2.len)
	{
		sample_radius2.resize(1);
		dev_sample_radius2 = sample_radius2.pdata;
	}
	const mxArray* lips_sample_points = prhs[1];
	nl = mxGetN(lips_sample_points);
	vec lsp;
	lsp.resize(nl * 3);
	cudaMemcpyAsync(lsp.pdata, mxGetDoubles(lips_sample_points), 3 * nl * sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_sample_radius2, mxGetDoubles(prhs[2]), sizeof(real), cudaMemcpyHostToDevice);


	mat temp_L1(5 * nl, nvnf);
	L1.resize(5 * nl, nvar);
	Ngamma.resize(nl, ne);
	temp1.resize(5 * nl, 3);
	temp2.resize(5 * nl, 3);
	coef_vH.resize(nl * 3);
	coef_NH.resize(nl * 2);

	real* H_phi_out = temp_L1.pdata;
	real* H_psi_out = H_phi_out + 5 * nl * nv;
	//Jpt.zero_fill();
	temp_L1.zero_fill();

	green_coords_3d_urago3_hessian_vectorized2 << <nl, threadsPerBlock >> > (cageT_in.pdata, nf, x1.pdata, x2.pdata, x3.pdata, N.pdata, nn.pdata, dx1.pdata, dx2.pdata, dx3.pdata, l1a.pdata, l2a.pdata, l3a.pdata,
		cNdx1.pdata, cNdx2.pdata, cNdx3.pdata, lsp.pdata, nl, H_phi_out, H_psi_out);

	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, 5 * nl, nvar, nvnf, dev_one, temp_L1.pdata, 5 * nl, ProjMat.pdata, nvnf, dev_zero, L1.pdata, 5 * nl);

	Lipschitz_Kernel_Matrix_Ngamma << <nl, threadsPerBlock >> > (cageX_in.pdata, nv, cageT_in.pdata, Half_edge_Info.pdata, ne, lsp.pdata, nl, dev_sample_radius2, Ngamma.pdata);
}


void deform(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs > 5) mexErrMsgTxt("ProjHarmonicMap('deform', numIter, variable, p2pDsts, qi);");
	const int numIter = mxGetScalar(prhs[1]);
	const mxArray* p2pDsts = prhs[3];
	if (numIter > 5) mexErrMsgTxt("Number of iteration must be less than 5!");
	const int numq = mxGetN(p2pDsts);

	if (nrhs == 5)
	{
		const mxArray* qi = prhs[4];
		vec q_in(numq * 3);
		cudaMemcpyAsync(q_in.pdata, mxGetDoubles(qi), numq * 3 * sizeof(real), cudaMemcpyHostToDevice);
		p2pDsts_in.resize(numq * 3);
		mat temp_p2p_matrix;
		temp_p2p_matrix.resize(numq, nvnf, false);
		real* phi_out = temp_p2p_matrix.pdata;
		real* psi_out = phi_out + numq * nv;
		temp_p2p_matrix.zero_fill();
		green_coords_3d_urago3_vectorized2 << <numq, threadsPerBlock >> > (cageT_in.pdata, nf, x1.pdata, x2.pdata, x3.pdata, N.pdata, nn.pdata, dx1.pdata, dx2.pdata, dx3.pdata, l1a.pdata, l2a.pdata, l3a.pdata, q_in.pdata, numq,
			phi_out, psi_out);

		p2p_matrix.resize(numq, nvar, false);
		gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, numq, nvar, nvnf, dev_one, temp_p2p_matrix.pdata, numq, ProjMat.pdata, nvnf, dev_zero, p2p_matrix.pdata, numq);
		
	}
	cudaMemcpyAsync(p2pDsts_in.pdata, mxGetDoubles(p2pDsts), 3 * numq * sizeof(real), cudaMemcpyHostToDevice);
	gemm(hdl, CUBLAS_OP_T, CUBLAS_OP_T, nvar, 3, numq, dev_w_p2p, p2p_matrix.pdata, numq, p2pDsts_in.pdata, 3, dev_zero, b.pdata, nvar);
	const mxArray* variable = prhs[2];
	cudaMemcpyAsync(z.pdata, mxGetDoubles(variable), nvar * 3 * sizeof(real), cudaMemcpyHostToDevice);
	gemm(hdl, CUBLAS_OP_T, CUBLAS_OP_N, nvar, nvar, numq, dev_w_p2p, p2p_matrix.pdata, numq, p2p_matrix.pdata, numq, dev_zero, M.pdata, nvar);
	geam(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, nvar, dev_one, M.pdata, nvar, dev_w_smooth, smooth_energy_h.pdata, nvar, M.pdata, nvar);
	for (int i = 0; i < numIter; i++)
	{

		ls = dInfo.pdata + 8 * i;
		e = ls + 1;
		bounds = e + 2;
		delta_norm = bounds + 4;
		HM.update_fill_derivatives();		
		HM.solve_linear_system();
		HM.linesearch();
		if (showTimeDetails)
		{
			HM.show_times();
		}
	}
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nd, 3, nvar, dev_one, D.pdata, nd, z.pdata, nvar, dev_zero, viewY.pdata, nd);

	plhs[0] = mxCreateDoubleMatrix(nvar, 3, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(nd, 3, mxREAL);	 
	plhs[2] = mxCreateDoubleMatrix(8, numIter, mxREAL);

	cudaMemcpyAsync(mxGetDoubles(plhs[0]), z.pdata, nvar * 3 * sizeof(real), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(mxGetDoubles(plhs[1]), viewY.pdata, nd * 3 * sizeof(real), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(mxGetDoubles(plhs[2]), dInfo.pdata, 8 * numIter * sizeof(real), cudaMemcpyDeviceToHost);
	if (nlhs = 4)
	{
		gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvnf, 3, nvar, dev_one, ProjMat.pdata, nvnf, z.pdata, nvar, dev_zero, newCageCoef.pdata, nvnf);
		plhs[3] = mxCreateDoubleMatrix(nvnf, 3, mxREAL);
		cudaMemcpyAsync(mxGetDoubles(plhs[3]), newCageCoef.pdata, 3 * nvnf * sizeof(real), cudaMemcpyDeviceToHost);
	}
}

void set_smooth_weight(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs != 2) mexErrMsgTxt("ProjHarmonicMap('set_smooth_weight', w_smooth);");
	real h_w_smooth = mxGetScalar(prhs[1]);
	cudaMemcpy(dev_w_smooth, &h_w_smooth, sizeof(real), cudaMemcpyHostToDevice);
}

void set_p2p_weight(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs != 2) mexErrMsgTxt("ProjHarmonicMap('set_p2p_weight', w_p2p);");
	real h_w_p2p = mxGetScalar(prhs[1]);
	cudaMemcpy(dev_w_p2p, &h_w_p2p, sizeof(real), cudaMemcpyHostToDevice);
}

void set_epsilon(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs != 2) mexErrMsgTxt("ProjHarmonicMap('set_epsilon', epsilon);");
	real eps = mxGetScalar(prhs[1]);
	cudaMemcpy(dev_sample_radius, &eps, sizeof(real), cudaMemcpyHostToDevice);
}

void set_DOF(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (nrhs != 2) mexErrMsgTxt("ProjHarmonicMap('set_DOF', DOFs);");
	nvar = mxGetScalar(prhs[1]);
}


#ifdef _DEBUG
void main()
{
	if (!initialized)
	{
		initialized = 1;
		createHdl();
		//mexAtExit(cleanup);
	}
	getMatEngine().connect("");
	{
		matlabEval("I = ProjHmDeformer", false);
		matlabEval("\
		cageX = gather(cages{1}.x)';\
		cageT = int32(gather(cages{1}.t) - 1)';\
		x = gather(ProjHmDeformer.x)';\
		tboundsampling = I.lipschitz_sample_points';\
		isometric_sample_points = gather(I.isometric_sample_points)';\
		ipara = struct('epsilon', gather(I.epsilon_iso), 'w_p2p', 1e8, 'epsilon2', I.epsilon_lips, 'w_smooth', I.smooth_weight, 'hess_sample_step_size', 150, 'energy_type', 1);\
		numIter = 1;\
		p2pDsts_In = p2pDsts';\
		z = gather(I.variable);\
		qi = gather(I.x(view_P2PVtxIds, :)');\
		", false);
		matlabEval("nv = size(cages{1}.x, 1);\
			nt = size(cages{1}.t, 1);\
			Half_edge_spmat = sparse(cages{1}.t(:, [1, 2, 3]), cages{1}.t(:, [2, 3, 1]), repmat((1:nt)',1,3), nv, nv);\
			[Half_edge1, Half_edge2, FaceId1] = find(tril(Half_edge_spmat));\
			[~, ~, FaceId2] = find(tril(Half_edge_spmat'));\
			I.edgeInfo = [Half_edge1, Half_edge2, FaceId1, FaceId2];\
			", "false");
		matlabEval("edgeInfo = int32(ProjHmDeformer.edgeInfo - 1)';", false);
		/*test for set_DOFs*/
		const mxArray* prhs0[2];
		prhs0[0] = NULL;
		prhs0[1] = getMatEngine().getVariable("DOFs");
		set_DOFs(0, NULL, 2, prhs0);

		/*test for set_model*/
		/*int nrhs = 5;*/
		const mxArray* prhs1[5];
		mxArray* plhs1[1];
		prhs1[0] = NULL;
		prhs1[1] = getMatEngine().getVariable("cageX");
		prhs1[2] = getMatEngine().getVariable("cageT");
		prhs1[3] = getMatEngine().getVariable("edgeInfo");
		prhs1[4] = getMatEngine().getVariable("x");
		set_model(0, plhs1, 5, prhs1);
		getMatEngine().putVariable("Proj_coef", plhs1[0]);

		mxArray* D_out;
		D_out = mxCreateDoubleMatrix(nd, nvar, mxREAL);
		cudaMemcpyAsync(mxGetDoubles(D_out), D.pdata, nd * nvar * sizeof(real), cudaMemcpyDeviceToHost);
		getMatEngine().putVariable("D_out", D_out);

		mxArray* Kappa_out;
		Kappa_out = mxCreateDoubleMatrix(3 * ne, nvar, mxREAL);
		cudaMemcpyAsync(mxGetDoubles(Kappa_out), Kappa.pdata, 3 * ne * nvar * sizeof(real), cudaMemcpyDeviceToHost);
		getMatEngine().putVariable("Kappa_out", Kappa_out);


		mxArray* L_out;
		L_out = mxCreateDoubleMatrix(nvar, nvar, mxREAL);
		cudaMemcpyAsync(mxGetDoubles(L_out), smooth_energy_h.pdata, nvar * nvar * sizeof(real), cudaMemcpyDeviceToHost);
		getMatEngine().putVariable("L_out", L_out);

		if (D_out) mxDestroyArray(D_out);
		if (Kappa_out) mxDestroyArray(Kappa_out);
		if (L_out) mxDestroyArray(L_out);

		/*test for set_deformer*/
		/*int nrhs = 3*/
		const mxArray* prhs2[3];
		prhs2[0] = NULL;
		prhs2[1] = getMatEngine().getVariable("isometric_sample_points");
		prhs2[2] = getMatEngine().getVariable("ipara");
		set_deformer(0, NULL, 3, prhs2);

		mxArray* Jp_out;
		Jp_out = mxCreateDoubleMatrix(nvar, 3 * nj, mxREAL);
		cudaMemcpyAsync(mxGetDoubles(Jp_out), Jp.pdata, nvar * 3 * nj * sizeof(real), cudaMemcpyDeviceToHost);
		getMatEngine().putVariable("Jp_out", Jp_out);

		/*test for set_lips_type*/
		const mxArray* prhs3[4];
		prhs3[0] = NULL;
		prhs3[1] = getMatEngine().getVariable("lt");
		prhs3[2] = getMatEngine().getVariable("isometric_sample_points");
		prhs3[3] = getMatEngine().getVariable("eps");
		set_lips_type(0, NULL, 4, prhs3);

		mxArray* L1_out;
		L1_out = mxCreateDoubleMatrix(5 * nl, nvar, mxREAL);
		cudaMemcpyAsync(mxGetDoubles(L1_out), L1.pdata, 5 * nl * nvar * sizeof(real), cudaMemcpyDeviceToHost);
		getMatEngine().putVariable("L1_out", L1_out);

		mxArray* Ngamma_out;
		Ngamma_out = mxCreateDoubleMatrix(nl, ne, mxREAL);
		cudaMemcpyAsync(mxGetDoubles(Ngamma_out), Ngamma.pdata, nl * ne * sizeof(real), cudaMemcpyDeviceToHost);
		getMatEngine().putVariable("Ngamma_out", Ngamma_out);

		if (Jp_out) mxDestroyArray(Jp_out);
		if (L1_out) mxDestroyArray(L1_out);
		if (Ngamma_out) mxDestroyArray(Ngamma_out);

		/*test for deformer*/
		const mxArray* prhs4[5];
		/*int nrhs = 4*/
		prhs4[0] = NULL;
		prhs4[1] = getMatEngine().getVariable("numIter");
		prhs4[2] = getMatEngine().getVariable("z");
		prhs4[3] = getMatEngine().getVariable("p2pDsts_In");
		prhs4[4] = getMatEngine().getVariable("qi");
		mxArray* plhs4[3];

		deform(3, plhs4, 5, prhs4);
		getMatEngine().putVariable("z_next", plhs4[0]);
		getMatEngine().putVariable("viewY_out", plhs4[1]);
		getMatEngine().putVariable("dInfo", plhs4[2]);
		if (plhs4[0]) mxDestroyArray(plhs4[0]);
		if (plhs4[1]) mxDestroyArray(plhs4[1]);
		if (plhs4[2]) mxDestroyArray(plhs4[2]);



	}
	cleanup();
}

#else
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	if (!initialized)
	{
		initialized = 1;
		createHdl();
		mexAtExit(cleanup);
	}

	if (nrhs < 1) mexErrMsgTxt("[...] = ProjHarmonicMap(para, ...); \npara can be 'set_model', 'set_deformer', 'deform'");
	//std::u16string_view para = u"help";	//visual studio 2019 has some issue about string_view...
	if (!mxIsChar(prhs[0])) mexErrMsgTxt("parameter should be a char array.");
	size_t len = mxGetNumberOfElements(prhs[0]) + 1;
	std::vector<char> str(len);
	mxGetString(prhs[0], str.data(), len);
	std::string para(str.cbegin(), str.cend() - 1);
	if (!para.compare("set_DOF"))
	{
		set_DOF(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("set_model"))
	{
		set_model(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("set_deformer"))
	{
		set_deformer(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("deform"))
	{
		deform(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("set_solver"))
	{
		set_solver(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("set_sampling_points"))
	{
		set_sampling_points(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("set_smooth_weight"))
	{
		set_smooth_weight(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("set_p2p_weight"))
	{
		set_p2p_weight(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("set_epsilon"))
	{
		set_epsilon(nlhs, plhs, nrhs, prhs);
	}
	else if (!para.compare("set_ls_step_size"))
	{
		set_ls_step_size(nlhs, plhs, nrhs, prhs);
	}
	else
	{
		mexErrMsgTxt("[...] = ProjHarmonicMap(para, ...); \npara can be 'set_model', 'set_deformer', 'deform'");
	}
}
#endif
