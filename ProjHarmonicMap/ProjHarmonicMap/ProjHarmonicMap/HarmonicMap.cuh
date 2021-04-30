#pragma once
#include"svd3_cuda.h"
//#include"estimate_sigma_value_real_time.h"
#include <assert.h>

#include"cub\block\block_reduce.cuh"

#include"utils.cuh"
#include"cuVec3.cuh"

#include "compute_green_coordinate.cuh"
#include "compute_lipschitz.cuh"

#include "math_constants.h"
#include "cub/device/device_select.cuh"

using vec = cuVector<real>;
using mat = cuMatrix<real>;


static cublasHandle_t hdl = 0;
static cudaEvent_t times[9];
static vec constants;
static real* dev_one = NULL;
static real* dev_zero = NULL;
static real* dev_minus_one = NULL;
static real* dev_two = NULL;
static real* dev_minus_two = NULL;
static real* dev_half = NULL;

/*static value for cage geometry*/
static vec cageX_in;
static cuVector<int> cageT_in;
static int nv = 0;
static int nf = 0;
static int ne = 0;
static int nd = 0;
static vec x1, x2, x3, N, nn, dx1, dx2, dx3, l1a, l2a, l3a;
static vec cNdx1, cNdx2, cNdx3;
static cuVector<int> Half_edge_Info;
static mat D;
static mat Kappa;

static vec viewY;

/*static value for deformer*/
static int initialized = 0;
static int showTimeDetails = 1;
static int nj = 0;
static int nh = 0;
static int nl = 0;
static int nvar = 0;
static int ss = 100;
static int et = 1;
static vec para;
static real* dev_sample_radius = NULL;
static real* dev_w_p2p = NULL;
static real* dev_w_smooth = NULL;
static real* dev_h_rate = NULL;
static real* dev_ls_step_size = NULL;
static real* lips_constant;
static vec sample_radius2;
static real* dev_sample_radius2 = NULL;


static mat p2p_matrix;
static mat Jp;
static cuMatrix<float> JpSp;
static mat L1;
static mat Ngamma;
//static mat Gamma;

static mat J, U, V, energy_h, smooth_energy_h, M, temp1, temp2;
static cuMatrix<float> temp;
static int solver_type;
static vec S, delta, energy_g, p2pDsts_in, b, scalars4linesearch, dInfo;
static cuVector<float> C;
static vec z, eq_g;
vec isp;

static cuVector<int> flip_point_idx;
static vec adaptive_sampling_position;
static int max_number_adaptive_sampling = 1e5;
static cuVector<int> max_idx_array;
static cuVector<int> d_temp_storage;
static cuVector<unsigned char> flag_list_of_failed;
static vec J_tmp_storage;
static vec H_norm_storage;
static cuVector<int> num_flip_points;
static vec tmp_s3;
static cuVector<int> adaptive_sampling_father_knot_idx;
static vec bd;
static bool sig3_low_bound_flag = true;




static real* dot_delta_g = NULL;
static real* q_coef = NULL;
static real* ls = NULL;
static real* e = NULL;
static real* bounds = NULL;
static real* delta_norm = NULL;

static vec dis_energy_gradient_resp2J;
static cuVector<float> Q;	//eigen vectors of isometric energy hessian matrix

static cuVector<const float*> DevArrayQ, DevArrayJp;
static cuVector<float*> DevArrayC;

static cuVector<float> dev_const_S;
static cuVector<float> eh_temp;

static vec coef_vH, coef_NH, ka_norm;

static mat ProjMat;
static int nvnf;
static vec Proj_z, Proj_delta;





#if defined(_DEBUG)
#undef printf
__device__ void printfv3(const real* v)
{
	printf("%lf, %lf, %lf\n", v[0], v[1], v[2]);

}
#endif









template<class R>
__global__ void fill_up_symmetric_matrix(R* A, const int m, cublasFillMode_t uplo)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= m * m)
		return;
	int i = tid / m;
	int j = tid - i * m;
	if (i >= j)
		return;
	switch (uplo)
	{
		case CUBLAS_FILL_MODE_LOWER:
			A[i + j * m] = A[j + i * m];
			break;
		case CUBLAS_FILL_MODE_UPPER:
			A[j + i * m] = A[i + j * m];
			break;
	}
}




template<class R>
__global__ void precompute_cageGeometry(const R* cageX, const int* cageT, const int nv, const int nf, R* x1, R* x2, R* x3, R* N, R* nn, R* dx1, R* dx2, R* dx3, R* l1a, R* l2a, R* l3a, R* cNdx1 = NULL, R* cNdx2 = NULL, R* cNdx3 = NULL)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nf)
	{
		cageT += 3 * tid;
		x1 += 3 * tid;
		x2 += 3 * tid;
		x3 += 3 * tid;
		N += 3 * tid;
		nn += tid;
		dx1 += 3 * tid;
		dx2 += 3 * tid;
		dx3 += 3 * tid;
		l1a += tid;
		l2a += tid;
		l3a += tid;
		int i1 = cageT[0]; int i2 = cageT[1]; int i3 = cageT[2];
		copyv3(cageX + i1 * 3, x1);
		copyv3(cageX + i2 * 3, x2);
		copyv3(cageX + i3 * 3, x3);
		minusv3(x2, x3, dx1);
		minusv3(x3, x1, dx2);
		minusv3(x1, x2, dx3);
		crossv3(dx2, dx3, N);
		nn[0] = normv3(N);
		normalizedv3(N);
		l1a[0] = normv3(dx1);
		l2a[0] = normv3(dx2);
		l3a[0] = normv3(dx3);

		if (cNdx1)
		{
			cNdx1 += 3 * tid;
			cNdx2 += 3 * tid;
			cNdx3 += 3 * tid;
			crossv3(N, dx1, cNdx1);
			crossv3(N, dx2, cNdx2);
			crossv3(N, dx3, cNdx3);
		}
	}
}



template<class R>
__global__ void green_coords_3d_urago3_vectorized2(const int* cageT, const int nf, const R* x1, const R* x2, const R* x3, const R* N, const R* nn, const R* dx1, const R* dx2, const R* dx3, const R* l1a, const R* l2a, const R* l3a, const R* p_in, const int np,
	R* phi_out, R* psi_out)
{
	int n = ceil(R(nf) / threadsPerBlock);
	int pid = blockIdx.x;
	if (pid < np)
	{
		for (int i = 0; i < n; i++)
		{
			int fid = threadIdx.x + i * threadsPerBlock;
			if (fid < nf)
			{
				int vid1 = cageT[3 * fid + 0];
				int vid2 = cageT[3 * fid + 1];
				int vid3 = cageT[3 * fid + 2];

				R* PSI = NULL;
				R PHI[3];

				PSI = psi_out + fid * np + pid;
				green_coords_3d_urago3(x1 + 3 * fid, x2 + 3 * fid, x3 + 3 * fid, N + 3 * fid, nn + fid, dx1 + 3 * fid, dx2 + 3 * fid, dx3 + 3 * fid, l1a + fid, l2a + fid, l3a + fid, p_in + pid * 3, PHI, PSI);

				atomicAdd(phi_out + vid1 * np + pid, PHI[0]);
				atomicAdd(phi_out + vid2 * np + pid, PHI[1]);
				atomicAdd(phi_out + vid3 * np + pid, PHI[2]);
			}
		}
	}
}

template<class R>
__global__ void green_coords_3d_urago3_gradient_vectorized2(const int* cageT, const int nf, const R* x1, const R* x2, const R* x3, const R* N, const R* nn, const R* dx1, const R* dx2, const R* dx3, const R* l1a, const R* l2a, const R* l3a,
	const R* p_in, const int np, R* J_phi_out, R* J_psi_out)
{
	int n = ceil(R(nf) / threadsPerBlock);
	int pid = blockIdx.x;
	if (pid < np)
	{
		for (int i = 0; i < n; i++)
		{
			int fid = threadIdx.x + i * threadsPerBlock;
			if (fid < nf)
			{
				int vid1 = cageT[3 * fid + 0];
				int vid2 = cageT[3 * fid + 1];
				int vid3 = cageT[3 * fid + 2];

				R* J_psi = NULL;
				R J_phi[9];
				J_psi = J_psi_out + fid * 3 * np + 3 * pid;

				green_coords_3d_urago3_gradient(x1 + 3 * fid, x2 + 3 * fid, x3 + 3 * fid, N + 3 * fid, nn + fid, dx1 + 3 * fid, dx2 + 3 * fid, dx3 + 3 * fid, l1a + fid, l2a + fid, l3a + fid, p_in + pid * 3, J_phi, J_psi);
				for (int j = 0; j < 3; j++)
				{
					atomicAdd(J_phi_out + vid1 * 3 * np + 3 * pid + j, J_phi[j]);
					atomicAdd(J_phi_out + vid2 * 3 * np + 3 * pid + j, J_phi[3 + j]);
					atomicAdd(J_phi_out + vid3 * 3 * np + 3 * pid + j, J_phi[6 + j]);
				}

			}
		}
	}
}


/*
template<class R>
__host__ void compute_high_res_sampling_points_green_coords_gradient(const R* p_in, const int np, R* Jp) // compute 2X-res sampling points gradient 
{
	vec h_res_p_in;
	h_res_p_in.resize(np * 3);
	vec temp_r;
	temp_r.resize(np);
	temp_r.set_value(dev_sample_radius);
	mat Jpt(3 * np, nvar);
	R* Jphi = Jpt.pdata;
	R* Jpsi = Jphi + 3 * np * nv;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				// each point in p_in would splits to 8 points
				cudaMemcpy(h_res_p_in.pdata, p_in, 3 * np * sizeof(R), cudaMemcpyDeviceToDevice);
				if (i)
					axpy(hdl, np, dev_one, temp_r.pdata, 1, h_res_p_in.pdata, 3);
				if (j)
					axpy(hdl, np, dev_one, temp_r.pdata, 1, h_res_p_in.pdata + 1, 3);
				if (k)
					axpy(hdl, np, dev_one, temp_r.pdata, 1, h_res_p_in.pdata + 2, 3);
				Jpt.zero_fill();

				green_coords_3d_urago3_gradient_vectorized2<<<np, threadsPerBlock>>>(cageT_in.pdata, nf, x1.pdata, x2.pdata, x3.pdata, N.pdata, nn.pdata, dx1.pdata, dx2.pdata, dx3.pdata, l1a.pdata, l2a.pdata, l3a.pdata, p_in, np, Jphi, Jpsi);
				
				int n = i * 4 + j * 2 + k;
				geam(hdl, CUBLAS_OP_T, CUBLAS_OP_N, nvar, 3 * np, dev_one, Jpt.pdata, 3 * np, dev_zero, NULL, nvar, Jp + n * 3 * np * nvar, nvar);
			}
		}
	}
}*/

template<class R>
__global__ void green_coords_3d_urago3_hessian_vectorized2(const int* cageT, const int nf, const R* x1, const R* x2, const R* x3, const R* N, const R* nn, const R* dx1, const R* dx2, const R* dx3, const R* l1a, const R* l2a, const R* l3a,
	const R* cNdx1, const R* cNdx2, const R* cNdx3, const R* p_in, const int np, R* H_phi_out, R* H_psi_out)
{
	int n = ceil(R(nf) / threadsPerBlock);
	int pid = blockIdx.x;
	if (pid < np)
	{
		for (int i = 0; i < n; i++)
		{
			int fid = threadIdx.x + i * threadsPerBlock;
			if (fid < nf)
			{

				int vid1 = cageT[3 * fid + 0];
				int vid2 = cageT[3 * fid + 1];
				int vid3 = cageT[3 * fid + 2];

				R* H_psi = NULL;
				R H_phi[15];
				H_psi = H_psi_out + fid * 5 * np + pid * 5;
				
				green_coords_3d_urago3_hessian(x1 + 3 * fid, x2 + 3 * fid, x3 + 3 * fid, N + 3 * fid, nn + fid, dx1 + 3 * fid, dx2 + 3 * fid, dx3 + 3 * fid, l1a + fid, l2a + fid, l3a + fid, cNdx1 + 3 * fid, cNdx2 + 3 * fid, cNdx3 + 3 * fid, p_in + pid * 3, H_phi, H_psi);

				for (int j = 0; j < 5; j++)
				{
					
					atomicAdd(H_phi_out + vid1 * 5 * np + 5 * pid + j, H_phi[j]);
					atomicAdd(H_phi_out + vid2 * 5 * np + 5 * pid + j, H_phi[5 + j]);
					atomicAdd(H_phi_out + vid3 * 5 * np + 5 * pid + j, H_phi[10 + j]);
				}
			}
		}
	}
}





template<class R>
__global__ void compute_isometric_energy_gradient(const R* u, const R* v, const R* s, const int nj, R* g_diagscales, int et)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nj)
	{
		int i9 = 9 * tid;
		int i3 = 3 * tid;
		//const R* u_1 = u + i9;	const R* u_2 = u + i9 + 3;	const R* u_3 = u + i9 + 6;
		//const R* v_1 = v + i9;	const R* v_2 = v + i9 + 3;	const R* v_3 = v + i9 + 6;
		R u11, u12, u13, u21, u22, u23, u31, u32, u33;
		R v11, v12, v13, v21, v22, v23, v31, v32, v33;
		u11 = u[i9];		u12 = u[i9 + 3];	u13 = u[i9 + 6];
		u21 = u[i9 + 1];	u22 = u[i9 + 4];	u23 = u[i9 + 7];
		u31 = u[i9 + 2];	u32 = u[i9 + 5];	u33 = u[i9 + 8];

		v11 = v[i9];		v12 = v[i9 + 3];	v13 = v[i9 + 6];
		v21 = v[i9 + 1];	v22 = v[i9 + 4];	v23 = v[i9 + 7];
		v31 = v[i9 + 2];	v32 = v[i9 + 5];	v33 = v[i9 + 8];

		R sigma1 = s[i3];	R sigma2 = s[i3 + 1];	R sigma3 = s[i3 + 2];
		R partial_sigma1, partial_sigma2, partial_sigma3;
		if (et == 1) //SD energy
		{
			partial_sigma1 = 2 * (sigma1 - 1 / sigma1 / sigma1 / sigma1);
			partial_sigma2 = 2 * (sigma2 - 1 / sigma2 / sigma2 / sigma2);
			partial_sigma3 = 2 * (sigma3 - 1 / sigma3 / sigma3 / sigma3);
		}
		else if (et == 2) //arap energy
		{
			partial_sigma1 = 2 * (sigma1 - 1);
			partial_sigma2 = 2 * (sigma2 - 1);
			partial_sigma3 = 2 * (sigma3 - 1);
		}
		/*A1*/
		g_diagscales[i3] = (v11 * u11 * partial_sigma1 + v12 * u12 * partial_sigma2 + v13 * u13 * partial_sigma3) / nj;
		g_diagscales[i3 + 1] = (v11 * u21 * partial_sigma1 + v12 * u22 * partial_sigma2 + v13 * u23 * partial_sigma3) / nj;
		g_diagscales[i3 + 2] = (v11 * u31 * partial_sigma1 + v12 * u32 * partial_sigma2 + v13 * u33 * partial_sigma3) / nj;

		/*A2*/
		g_diagscales[i3 + 3 * nj] = (v21 * u11 * partial_sigma1 + v22 * u12 * partial_sigma2 + v23 * u13 * partial_sigma3) / nj;
		g_diagscales[i3 + 3 * nj + 1] = (v21 * u21 * partial_sigma1 + v22 * u22 * partial_sigma2 + v23 * u23 * partial_sigma3) / nj;
		g_diagscales[i3 + 3 * nj + 2] = (v21 * u31 * partial_sigma1 + v22 * u32 * partial_sigma2 + v23 * u33 * partial_sigma3) / nj;

		/*A3*/
		g_diagscales[i3 + 6 * nj] = (v31 * u11 * partial_sigma1 + v32 * u12 * partial_sigma2 + v33 * u13 * partial_sigma3) / nj;
		g_diagscales[i3 + 6 * nj + 1] = (v31 * u21 * partial_sigma1 + v32 * u22 * partial_sigma2 + v33 * u23 * partial_sigma3) / nj;
		g_diagscales[i3 + 6 * nj + 2] = (v31 * u31 * partial_sigma1 + v32 * u32 * partial_sigma2 + v33 * u33 * partial_sigma3) / nj;
	}

}

/*C=kron(A,B)'=kron(A',B')    A,B,C are 3-by-3 matrices*/
template<class R>
__device__ void kron3T(const R* A, const R* B, R* C)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					C[(3 * j + l) * 9 + 3 * i + k] = A[i * 3 + j] * B[3 * k + l];
				}
			}
		}
	}
}


template<class R>
__global__ void compute_isometric_energy_hessian(const R* u, const R* v, const R* s, int numds, int ss, float* Q, int et)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < numds)
	{
		int id = tid * ss;
		int i9 = 9 * id;
		int i3 = 3 * id;

		R sigma1 = s[i3];	R sigma2 = s[i3 + 1];	R sigma3 = s[i3 + 2];

		/*
			hess_J(E)=Q*|ev1									|*Q'
						|	ev2									|
						|		ev3								|
						|			2*a1 2*b1					|
						|			2*b1 2*a1					|
						|					 2*a2 2*b2			|
						|					 2*b2 2*a2			|
						|							  2*a3 2*b3 |
						|							  2*b3 2*a3 |
		*/
		R ev1, ev2, ev3, a1, b1, a2, b2, a3, b3;
		if (et == 1)
		{
			R s1s2 = sigma1 * sigma2;
			R s1s3 = sigma1 * sigma3;
			R s2s3 = sigma2 * sigma3;
			ev1 = 2 * (1 + 3 / sigma1 / sigma1 / sigma1 / sigma1);
			ev2 = 2 * (1 + 3 / sigma2 / sigma2 / sigma2 / sigma2);
			ev3 = 2 * (1 + 3 / sigma3 / sigma3 / sigma3 / sigma3);
			a1 = 1 + 1 / s1s2 / s1s2;
			b1 = (sigma1 * sigma1 + sigma2 * sigma2) / s1s2 / s1s2 / s1s2;
			a2 = 1 + 1 / s1s3 / s1s3;
			b2 = (sigma1 * sigma1 + sigma3 * sigma3) / s1s3 / s1s3 / s1s3;
			a3 = 1 + 1 / s2s3 / s2s3;
			b3 = (sigma2 * sigma2 + sigma3 * sigma3) / s2s3 / s2s3 / s2s3;
		}
		else if (et == 2)
		{
			ev1 = 2;
			ev2 = 2;
			ev3 = 2;
			a1 = 1 - 1 / (sigma1 + sigma2);
			b1 = 1 / (sigma1 + sigma2);
			a2 = 1 - 1 / (sigma1 + sigma3);
			b2 = 1 / (sigma1 + sigma3);
			a3 = 1 - 1 / (sigma2 + sigma3);
			b3 = 1 / (sigma2 + sigma3);
		}


		R s_eig[9];
		s_eig[0] = sqrt(ev1);
		s_eig[1] = sqrt(ev2);
		s_eig[2] = sqrt(ev3);
		s_eig[3] = sqrt(a1 + b1);
		s_eig[4] = sqrt((a1 - b1) > 0 ? (a1 - b1) : 0);
		s_eig[5] = sqrt(a2 + b2);
		s_eig[6] = sqrt((a2 - b2) > 0 ? (a2 - b2) : 0);
		s_eig[7] = sqrt(a3 + b3);
		s_eig[8] = sqrt((a3 - b3) > 0 ? (a3 - b3) : 0); 

		float* Mn = Q + tid * 81;
		R temp[81];
		kron3T(v + i9, u + i9, temp);
		for (int j = 0; j < 9; j++)
		{
			Mn[j * 9] = temp[j * 9] * s_eig[0];
			Mn[j * 9 + 1] = temp[j * 9 + 4] * s_eig[1];
			Mn[j * 9 + 2] = temp[j * 9 + 8] * s_eig[2];
			Mn[j * 9 + 3] = (temp[j * 9 + 1] + temp[j * 9 + 3]) * s_eig[3];
			Mn[j * 9 + 4] = (temp[j * 9 + 1] - temp[j * 9 + 3]) * s_eig[4];
			Mn[j * 9 + 5] = (temp[j * 9 + 2] + temp[j * 9 + 6]) * s_eig[5];
			Mn[j * 9 + 6] = (temp[j * 9 + 2] - temp[j * 9 + 6]) * s_eig[6];
			Mn[j * 9 + 7] = (temp[j * 9 + 5] + temp[j * 9 + 7]) * s_eig[7];
			Mn[j * 9 + 8] = (temp[j * 9 + 5] - temp[j * 9 + 7]) * s_eig[8];
			if (et == 3)
			{
				R sd = sigma1 * sigma1 + 1 / sigma1 / sigma1 + sigma2 * sigma2 + 1 / sigma2 / sigma2 + sigma3 * sigma3 + 1 / sigma3 / sigma3;
				R esd = exp(sd);
				R ee = 2 * sqrt(esd / 3);
				R d = ee * (temp[j * 9] * (sigma1 - 1 / sigma1 / sigma1 / sigma1) + temp[j * 9 + 4] * (sigma2 - 1 / sigma2 / sigma2 / sigma2) + temp[j * 9 + 8] * (sigma3 - 1 / sigma3 / sigma3 / sigma3));
				Mn[j * 9] += d;
				Mn[j * 9 + 1] += d;
				Mn[j * 9 + 2] += d;
			}
		}
	}
}




template<class R>
__global__ void svd_jacobian_matrix(const R* J, const int n, R* U = NULL, R* S = NULL, R* V = NULL, R* e = NULL, const int et = 1, const R* dJ = NULL, R* ls = 0)	// return the svd result of [u,s,v] = svd(J - ls * dJ), and energy e = symDirichlet(s)  
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	R ep = 0;

	if (tid < n)
	{
		R J11, J12, J13, J21, J22, J23, J31, J32, J33;


		//input:J
		J11 = J[tid * 3];		J12 = J[tid * 3 + n * 3];		J13 = J[tid * 3 + n * 6];
		J21 = J[tid * 3 + 1];	J22 = J[tid * 3 + n * 3 + 1];	J23 = J[tid * 3 + n * 6 + 1];
		J31 = J[tid * 3 + 2];	J32 = J[tid * 3 + n * 3 + 2];	J33 = J[tid * 3 + n * 6 + 2];


		if (dJ)
		{
			R step = *ls;
			J11 -= dJ[tid * 3] * step;
			J12 -= dJ[tid * 3 + n * 3] * step;
			J13 -= dJ[tid * 3 + n * 6] * step;
			J21 -= dJ[tid * 3 + 1] * step;
			J22 -= dJ[tid * 3 + n * 3 + 1] * step;
			J23 -= dJ[tid * 3 + n * 6 + 1] * step;
			J31 -= dJ[tid * 3 + 2] * step;
			J32 -= dJ[tid * 3 + n * 3 + 2] * step;
			J33 -= dJ[tid * 3 + n * 6 + 2] * step;
		}


		R u11, u12, u13, u21, u22, u23, u31, u32, u33;
		R s11, s12, s13, s21, s22, s23, s31, s32, s33;
		R v11, v12, v13, v21, v22, v23, v31, v32, v33;

		svd(J11, J12, J13, J21, J22, J23, J31, J32, J33,
			u11, u12, u13, u21, u22, u23, u31, u32, u33,
			s11, s12, s13, s21, s22, s23, s31, s32, s33,
			v11, v12, v13, v21, v22, v23, v31, v32, v33);

		int bp = 9 * tid;
		if (U)
		{
			U[bp] = u11;		U[bp + 3] = u12;	U[bp + 6] = u13;
			U[bp + 1] = u21;	U[bp + 4] = u22;	U[bp + 7] = u23;
			U[bp + 2] = u31;	U[bp + 5] = u32;	U[bp + 8] = u33;

			S[3 * tid] = s11;	S[3 * tid + 1] = s22;	S[3 * tid + 2] = s33;

			V[bp] = v11;		V[bp + 3] = v12;	V[bp + 6] = v13;
			V[bp + 1] = v21;	V[bp + 4] = v22;	V[bp + 7] = v23;
			V[bp + 2] = v31;	V[bp + 5] = v32;	V[bp + 8] = v33;
		}
		if (et == 1)
			ep = s11 * s11 + s22 * s22 + s33 * s33 + 1 / s11 / s11 + 1 / s22 / s22 + 1 / s33 / s33 - 6;
		else if (et == 2)
			ep = (s11 - 1) * (s11 - 1) + (s22 - 1) * (s22 - 1) + (s33 - 1) * (s33 - 1);
		else if (et == 3)
			ep = exp(s11 * s11 + s22 * s22 + s33 * s33 + 1 / s11 / s11 + 1 / s22 / s22 + 1 / s33 / s33);

	}
	if (e)
	{
		using BlockReduceT = cub::BlockReduce<R, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
		__shared__ typename BlockReduceT::TempStorage temp_storage;

		R block_e = R(BlockReduceT(temp_storage).Sum(ep));
		if (threadIdx.x == 0)
			atomicAdd(e, block_e / n);
	}
}

template<class R>
__global__ void harmonic_linesearch_for_energy_decrease(const R* J, const int n, const R* dJ, const R* coef, const R* dot_delta_g, const R* delta_norm, R* e, const int et, R* ls/*, int enEvalsPerKernel = 4*/)
{
	const R ls_alpha = 0.2;
	const R ls_beta = 0.5;


	R* en = e + 1;
	auto fQuadEnergy = [&](R t) {return (coef[0] + coef[1] * t + coef[2] * t * t); };
	auto fQPEstim = [&](R t) {return (*e - ls_alpha * (*dot_delta_g) * t); };
	(*ls) = 1;
	do
	{
		*en = 0;
		svd_jacobian_matrix << <blockNum(n), threadsPerBlock >> > (J, n, (R*)NULL, (R*)NULL, (R*)NULL, en, et, dJ, ls);
		cudaDeviceSynchronize();
		*en += fQuadEnergy(*ls);
		(*ls) = (*ls) * ls_beta;
	} while (((*en) > fQPEstim(*ls)) && ((*ls) * 2 * (*delta_norm) > 1e-6));//wolfe condition

	(*ls) = (*ls) * 2;
	//*energy_next = E_next;
}


__global__ void generate_max_index_array(const int max_index, int* idx_array)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < max_index)
	{
		idx_array[tid] = tid;
	}
}

template<class R>
__host__ void compute_lips_coef_3(R* coef_vH, R* coef_NH, R* ka_norm, const R* L, const R* Ngamma, const R* Kappa, const R* z, const R* delta, R* tmp1, R* tmp2, const int np, const int ne, const int nvar)
{
	/*
	*	compute vertex Hessian coef:
	*		|vH| = norm(L(z - t*delta)) = sqrt(coef_vH[0] + coef_vH[1] * t + coef_vH[2] * t^2)
	*/
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, 5 * np, 3, nvar, dev_one, L, 5 * np, z, nvar, dev_zero, tmp1, 5 * np); // temp1 : H_n = L * z
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, 5 * np, 3, nvar, dev_one, L, 5 * np, delta, nvar, dev_zero, tmp2, 5 * np); // temp2 : deltaH = L * delta
	compute_vertex_Hessian_coef_array << < blockNum(np), threadsPerBlock >> > (coef_vH, tmp1, tmp2, np);

	if (!coef_NH)
		return;

	/*
	*	compute Nabla(H) coef:
	*		|NH| = |Ngamma| |K(z - t*delta)| <= |Ngamma| |K*z| + |Ngamma| |K*delta| t
	*/
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, 3 * ne, 3, nvar, dev_one, Kappa, 3 * ne, z, nvar, dev_zero, tmp1, 3 * ne); // temp1 : K_n = Kappa * z
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, 3 * ne, 3, nvar, dev_one, Kappa, 3 * ne, delta, nvar, dev_zero, tmp2, 3 * ne); // temp2 : deltaKappa = Kappa * delta
	compute_kappa_coef << <blockNum(ne), threadsPerBlock >> > (tmp1, tmp2, ka_norm, ne);
	gemv(hdl, CUBLAS_OP_N, np, ne, dev_one, Ngamma, np, ka_norm, 2, dev_zero, coef_NH, 2);
	gemv(hdl, CUBLAS_OP_N, np, ne, dev_one, Ngamma, np, ka_norm + 1, 2, dev_zero, coef_NH + 1, 2);

}

template<class R>
__device__ void compute_sigma_bound(R J11, R J12, R J13, R J21, R J22, R J23, R J31, R J32, R J33, R* all_bounds, R& min_positive_s3, R lips_constant, R r1, unsigned char* failed_list ,int tid, R threshold = 0)
{
	R u11, u12, u13, u21, u22, u23, u31, u32, u33;
	R s11, s12, s13, s21, s22, s23, s31, s32, s33;
	R v11, v12, v13, v21, v22, v23, v31, v32, v33;

	svd(J11, J12, J13, J21, J22, J23, J31, J32, J33,
		u11, u12, u13, u21, u22, u23, u31, u32, u33,
		s11, s12, s13, s21, s22, s23, s31, s32, s33,
		v11, v12, v13, v21, v22, v23, v31, v32, v33);
	all_bounds[0] = s11;
	all_bounds[1] = s11 + lips_constant * (r1);
	all_bounds[2] = exp(-s33);
	all_bounds[3] = exp(-s33 + lips_constant * (r1));
	min_positive_s3 = (s33 - lips_constant * (r1) > threshold) ? exp(-s33 + lips_constant * (r1)) : exp(-1.0);
	if (failed_list)
	{
		if ((s33 - lips_constant * (r1)) > threshold)
		{
			failed_list[tid] = 0;
		}
		else
		{
			failed_list[tid] = unsigned char(1);
		}
	}
}

template<class R>
__global__ void estim_distortion_bound(const R* J, const int n, const R r1, const R r2, const R* ls, R* lips_constant, R* bound, R* tmp_min_s3, unsigned char* failed_list, 
	const R* coef_vH, const R* coef_NH, R* dJ = NULL, R threshold = 0)	// 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	R all_bounds[4] = { 0, 0, 0, 0 };
	R min_positive_s3 = 0;
	R lips = 0;
	if (tid < n)
	{

		lips = sqrt(coef_vH[3 * tid] + (*ls) * coef_vH[3 * tid + 1] + (*ls) * (*ls) * coef_vH[3 * tid + 2]) + r2 * (coef_NH[tid * 2] + (*ls) * coef_NH[tid * 2 + 1]);


		R J11, J12, J13, J21, J22, J23, J31, J32, J33;


		//input:J
		J11 = J[tid * 3];		J12 = J[tid * 3 + n * 3];		J13 = J[tid * 3 + n * 6];
		J21 = J[tid * 3 + 1];	J22 = J[tid * 3 + n * 3 + 1];	J23 = J[tid * 3 + n * 6 + 1];
		J31 = J[tid * 3 + 2];	J32 = J[tid * 3 + n * 3 + 2];	J33 = J[tid * 3 + n * 6 + 2];


		if (dJ)
		{
			R step = *ls;
			J11 -= dJ[tid * 3] * step;
			J12 -= dJ[tid * 3 + n * 3] * step;
			J13 -= dJ[tid * 3 + n * 6] * step;
			J21 -= dJ[tid * 3 + 1] * step;
			J22 -= dJ[tid * 3 + n * 3 + 1] * step;
			J23 -= dJ[tid * 3 + n * 6 + 1] * step;
			J31 -= dJ[tid * 3 + 2] * step;
			J32 -= dJ[tid * 3 + n * 3 + 2] * step;
			J33 -= dJ[tid * 3 + n * 6 + 2] * step;
		}
		compute_sigma_bound(J11, J12, J13, J21, J22, J23, J31, J32, J33, all_bounds, min_positive_s3, lips, r1, failed_list, tid, threshold);
	}

	using BlockReduceT = cub::BlockReduce<R, threadsPerBlock, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
	__shared__ typename BlockReduceT::TempStorage  temp_storage;
	for (int i = 0; i < 4; i++) {
		R block_bound = R(BlockReduceT(temp_storage).Reduce(all_bounds[i], cub::Max()));
		if (threadIdx.x == 0 && bound)
			atomicMaxPositive(bound + i, block_bound);
		  __syncthreads();
	}
	
	R block_mps = R(BlockReduceT(temp_storage).Reduce(min_positive_s3, cub::Max()));
	if (threadIdx.x == 0 && tmp_min_s3)
		atomicMaxPositive(tmp_min_s3, block_mps);

}

template<class R>
__global__ void realtime_estim_distortion_bound(const R* J, const int n, const R r1, const R r2, const R* ls, R* bound, R* tmp_min_s3, unsigned char* failed_list,
	const int* adaptive_sampling_father_knot_idx, const R* coef_vH, const R* coef_NH, R threshold = 0)	// 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	R all_bounds[4] = { 0, 0, 0, 0 };
	R min_positive_s3 = 0;
	R lips = 0;

	if (tid < n)
	{
		int idx = adaptive_sampling_father_knot_idx[tid];
		lips = sqrt(coef_vH[3 * idx] + (*ls) * coef_vH[3 * idx + 1] + (*ls) * (*ls) * coef_vH[3 * idx + 2]) + r2 * (coef_NH[idx * 2] + (*ls) * coef_NH[idx * 2 + 1]);

		R J11, J12, J13, J21, J22, J23, J31, J32, J33;


		//input:J
		J11 = J[tid * 3];		J12 = J[tid * 3 + n * 3];		J13 = J[tid * 3 + n * 6];
		J21 = J[tid * 3 + 1];	J22 = J[tid * 3 + n * 3 + 1];	J23 = J[tid * 3 + n * 6 + 1];
		J31 = J[tid * 3 + 2];	J32 = J[tid * 3 + n * 3 + 2];	J33 = J[tid * 3 + n * 6 + 2];


		compute_sigma_bound(J11, J12, J13, J21, J22, J23, J31, J32, J33, all_bounds, min_positive_s3, lips, r1, failed_list, tid, threshold);
	}

	using BlockReduceT = cub::BlockReduce<R, threadsPerBlock, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
	__shared__ typename BlockReduceT::TempStorage  temp_storage;
	for (int i = 0; i < 4; i++) {
		R block_bound = R(BlockReduceT(temp_storage).Reduce(all_bounds[i], cub::Max()));
		if (threadIdx.x == 0)
			atomicMaxPositive(bound + i, block_bound);
		__syncthreads();
	}
	R block_mps = R(BlockReduceT(temp_storage).Reduce(min_positive_s3, cub::Max()));
	if (threadIdx.x == 0)
		atomicMaxPositive(tmp_min_s3, block_mps);

}


/*CUBLAS cannt be called in device function in CUDA 10*/
template<class R>
__global__ void mygemv(int m, int n, const R* alpha, const R* A, const R* x, const R* beta, R* y) // y = alpha*Ax + y
{
	int num = ceil(R(n) / threadsPerBlock);
	int i = blockIdx.x;
	R temp = 0;
	if (i < m)
	{
		for (int k = 0; k < num; k++)
		{
			int j = threadIdx.x + k * threadsPerBlock;
			if (j < n)
			{
				temp += A[j * m + i] * x[j];
			}
		}
	}

	using BlockReduceT = cub::BlockReduce<R, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
	__shared__ typename BlockReduceT::TempStorage temp_storage;
	R block_temp = R(BlockReduceT(temp_storage).Sum(temp));

	if (threadIdx.x == 0 && i < m)
	{
		y[i] = alpha[0] * block_temp + beta[0] * y[i];
	}

}

template<class R>
__global__ void compute_lipshitz_constant_of_map(int nl, int ne, int approx_lips, const R* Ngamma, const R* kappa, const R r, const R* vLip, R* Lips_constant)
{
	int num = ceil(R(ne) / threadsPerBlock);
	int i = blockIdx.x;
	R temp = 0;
	R block_temp = 0;
	if (i < nl)
	{
		if (!approx_lips)
		{
			for (int k = 0; k < num; k++)
			{
				int j = threadIdx.x + k * threadsPerBlock;
				if (j < ne)
				{
					temp += Ngamma[j * nl + i] * kappa[j];
				}
			}
			using BlockReduceT = cub::BlockReduce<R, threadsPerBlock, cub::BLOCK_REDUCE_RAKING>;
			__shared__ typename BlockReduceT::TempStorage temp_storage;
			block_temp = R(BlockReduceT(temp_storage).Sum(temp));
		}

		if (threadIdx.x == 0)
		{
			R lip = block_temp * r + vLip[i];
			atomicMaxPositive(Lips_constant, lip);
		}
	}
}



template<class R>
__global__ void compute_Jacobian_at_failed_points(const int* cageT, const int nf, const int nv, const R* x1, const R* x2, const R* x3, const R* N, const R* nn, const R* dx1, const R* dx2, const R* dx3, const R* l1a, const R* l2a, const R* l3a,
	const R* p_in, const int np, const R* z, const R* delta,  const R ls, R* J_tmp_storage)
{
	int n = ceil(R(nf) / threadsPerBlock);
	int pid = blockIdx.x;
	R Jblock[9] = { 0,0,0,0,0,0,0,0,0 };
	if (pid < np)
	{
		for (int i = 0; i < n; i++)
		{
			int fid = threadIdx.x + i * threadsPerBlock;
			if (fid < nf)
			{
				int vid1 = cageT[3 * fid + 0];
				int vid2 = cageT[3 * fid + 1];
				int vid3 = cageT[3 * fid + 2];

				R J_psi[3];
				R J_phi[9];
				

				green_coords_3d_urago3_gradient(x1 + 3 * fid, x2 + 3 * fid, x3 + 3 * fid, N + 3 * fid, nn + fid, dx1 + 3 * fid, dx2 + 3 * fid, dx3 + 3 * fid, l1a + fid, l2a + fid, l3a + fid, p_in + pid * 3, J_phi, J_psi);
				
				for (int j = 0; j < 3; j++)
					for (int k = 0; k < 3; k++)
					{
						Jblock[3 * k + j] += J_phi[j] * (z[vid1 + (nf + nv) * k] - ls * delta[vid1 + (nf + nv) * k]) + J_phi[3 + j] * (z[vid2 + (nf + nv) * k] - ls * delta[vid2 + (nf + nv) * k]) 
							+ J_phi[6 + j] * (z[vid3 + (nf + nv) * k] - ls * delta[vid3 + (nf + nv) * k]) + J_psi[j] * (z[nv + fid + (nf + nv) * k] - ls * delta[nv + fid + (nf + nv) * k]);
					}
				

			}
		}

		
		using BlockReduceT = cub::BlockReduce<R, threadsPerBlock, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
		__shared__ typename BlockReduceT::TempStorage temp_storage;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				R block_temp = R(BlockReduceT(temp_storage).Sum(Jblock[3 * i + j]));

				if (threadIdx.x == 0)
				{
					J_tmp_storage[3 * pid + i * 3 * np + j] = block_temp;
				}
				if (i * 3 + j < 8) __syncthreads();
			}
	}

}



/*subdivision one point to eight points*/
template<class R>
__global__ void high_res_sampling(R* next_x, const R* front_x, int* next_father_knot, const int* front_father_knot, const int* flip_point_idx, const int num_item, const R r)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < num_item)
	{
		R offset = r / 2;
		int idx = flip_point_idx[tid];
		next_x[tid * 24] = front_x[idx * 3] - offset;			next_x[tid * 24 + 1] = front_x[idx * 3 + 1] - offset;		next_x[tid * 24 + 2] = front_x[idx * 3 + 2] - offset;
		next_x[tid * 24 + 3] = front_x[idx * 3] - offset;		next_x[tid * 24 + 4] = front_x[idx * 3 + 1] - offset;		next_x[tid * 24 + 5] = front_x[idx * 3 + 2] + offset;
		next_x[tid * 24 + 6] = front_x[idx * 3] - offset;		next_x[tid * 24 + 7] = front_x[idx * 3 + 1] + offset;		next_x[tid * 24 + 8] = front_x[idx * 3 + 2] - offset;
		next_x[tid * 24 + 9] = front_x[idx * 3] - offset;		next_x[tid * 24 + 10] = front_x[idx * 3 + 1] + offset;		next_x[tid * 24 + 11] = front_x[idx * 3 + 2] + offset;
		next_x[tid * 24 + 12] = front_x[idx * 3] + offset;		next_x[tid * 24 + 13] = front_x[idx * 3 + 1] - offset;		next_x[tid * 24 + 14] = front_x[idx * 3 + 2] - offset;
		next_x[tid * 24 + 15] = front_x[idx * 3] + offset;		next_x[tid * 24 + 16] = front_x[idx * 3 + 1] - offset;		next_x[tid * 24 + 17] = front_x[idx * 3 + 2] + offset;
		next_x[tid * 24 + 18] = front_x[idx * 3] + offset;		next_x[tid * 24 + 19] = front_x[idx * 3 + 1] + offset;		next_x[tid * 24 + 20] = front_x[idx * 3 + 2] - offset;
		next_x[tid * 24 + 21] = front_x[idx * 3] + offset;		next_x[tid * 24 + 22] = front_x[idx * 3 + 1] + offset;		next_x[tid * 24 + 23] = front_x[idx * 3 + 2] + offset;


		for (int i = 0; i < 8; i++)
		{
			next_father_knot[tid * 8 + i] = front_father_knot[idx];
		}
	}
}




template<class R>
__global__ void harmonic_linesearch_for_locally_injective3
	(const R* J, const int nj, R* dJ, R* lips_constant, R* sample_radius, R* sample_radius2, const int nl, const R* coef_vH,
	const R* coef_NH, const R* coef_ka, const R* normdpp, R* ls, R* bounds, const int et, const R* z, const R* delta,
	R* adaptive_sampling_position, R* ls_step_size, unsigned char* failed_list, int* d_temp_storage, R* J_tmp_storage, R* H_norm_storage, int* num_flip_points, R* tmp_min_s3, R* bd, const int* max_idx_array, int* flip_point_idx,
	const int max_number_adaptive_sampling, int* adaptive_sampling_father_knot_idx,
	const int* cageT_in, const int nf, const int nv, const R* x1, const R* x2, const R* x3, const R* N, const R* nn, const R* dx1, const R* dx2, const R* dx3, const R* l1a, const R* l2a, const R* l3a,
	const R* cNdx1, const R* cNdx2, const R* cNdx3, bool sig3_low_bound_flag
	)
{
	R r1 = sqrt(3.0) / 2 * (*sample_radius);
	R r2 = sqrt(2.0) / 2 * (*sample_radius2);
	do
	{
		*num_flip_points = 0;
		*tmp_min_s3 = 0;
		R sig3_low_bound = 0;

		cudaMemsetAsync(bounds, 0, sizeof(R) * 4);
		lips_constant[0] = 0;		
		if (sig3_low_bound_flag)	//sig3 - Lp * r > 1e-4
		{
			sig3_low_bound = 1e-4;
		}
			
		estim_distortion_bound << <blockNum(nj), threadsPerBlock >> > (J, nj, r1, r2, ls, lips_constant, bounds, tmp_min_s3, failed_list, coef_vH, coef_NH, dJ, sig3_low_bound);
		cudaDeviceSynchronize();
		bounds[2] = -log(bounds[2]);
		bounds[3] = -log(bounds[3]);			
	
	
		if (bounds[3] > 0 || et == 2)
		{
			*ls = -(*ls); //return norm(t*delta)
			return;
		}
		else if ((*ls > *ls_step_size  || bounds[2] < 0))
		{
			*ls = *ls * 0.5;
		}
		else
		{
			int num_items = nj;
			R tmp_sp_radius = *sample_radius / 2;
			R tmp_r1 = r1 / 2;
			R tmp_r2 = r2 / 2;
			int it = 0;
			do
			{
				cudaMemsetAsync(bd, 0, sizeof(R) * 4);
				//cudaMemcpyAsync(flip_point_idx, max_idx_array, num_items * sizeof(int), cudaMemcpyDeviceToDevice);
				size_t	temp_storage_bytes = 0;
				cub::DeviceSelect::Flagged(NULL, temp_storage_bytes, max_idx_array, failed_list, flip_point_idx, num_flip_points, num_items);
				if (temp_storage_bytes < max_number_adaptive_sampling)
					cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, max_idx_array, failed_list, flip_point_idx, num_flip_points, num_items);
				else
					break;
				cudaDeviceSynchronize();
				if ((*num_flip_points) * 8 < max_number_adaptive_sampling)
				{
					high_res_sampling << <blockNum(*num_flip_points), threadsPerBlock >> > (adaptive_sampling_position + (it % 2) * max_number_adaptive_sampling * 3, 
																							adaptive_sampling_position + ((it + 1) % 2) * max_number_adaptive_sampling * 3,
																							adaptive_sampling_father_knot_idx + (it % 2) * max_number_adaptive_sampling,
																							adaptive_sampling_father_knot_idx + ((it + 1) % 2) * max_number_adaptive_sampling,
																							flip_point_idx , 
																							(*num_flip_points),
																							tmp_sp_radius);

					num_items = (*num_flip_points) * 8;
					compute_Jacobian_at_failed_points << < num_items, threadsPerBlock >> > (cageT_in, nf, nv, x1, x2, x3, N, nn, dx1, dx2, dx3, l1a, l2a, l3a,
						adaptive_sampling_position + (it % 2) * max_number_adaptive_sampling * 3, num_items, z, delta, *ls, J_tmp_storage);					
					realtime_estim_distortion_bound << <blockNum(num_items), threadsPerBlock >> > (J_tmp_storage, num_items, tmp_r1, r2, ls, bd, tmp_min_s3, failed_list,
						adaptive_sampling_father_knot_idx + (it % 2) * max_number_adaptive_sampling, coef_vH, coef_NH, sig3_low_bound);
					
					cudaDeviceSynchronize();
					bd[3] = -log(bd[3]);
					bd[2] = -log(bd[2]);
					if (bd[3] > sig3_low_bound)
					{
						*tmp_min_s3 = -log(*tmp_min_s3);
						bounds[3] = bd[3] > * tmp_min_s3 ? *tmp_min_s3 : bd[3];
						*ls = -(*ls); //return norm(t*delta)
						return;
					}
				}
				else
					break;
				it = (it + 1);
				tmp_sp_radius /= 2;
				tmp_r1 /= 2;
				tmp_r2 /= 2;

			} while (bd[2] > 0);

			*ls = *ls * 0.5;

		}
	} while ((*ls) * (*normdpp) > 1e-6);
	*ls = 0;
	cudaMemsetAsync(bounds, 0, sizeof(R) * 4);
	estim_distortion_bound << <blockNum(nj), threadsPerBlock >> > (J, nj, r1, r2, ls, lips_constant, bounds, tmp_min_s3, failed_list, coef_vH, coef_NH);
	cudaDeviceSynchronize();
	bounds[2] = -log(bounds[2]);
	bounds[3] = 1e-4;

}


template<class T1, class T2>
__global__ void convertType(T1* dst, T2* src, const int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n)
		dst[tid] = src[tid];
}

template<class R>
__global__ void setValue(R* A, R value, const int n, const int incA)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n)
		A[tid * incA] += value;
}

template<class R>
struct HarmonicMap
{



	HarmonicMap()
	{
	}
	~HarmonicMap()
	{
		/*cudaDeviceSynchronize();
		const char* headers[] = { "svd", "jacob diff", "fill gradient", "fill hessian part1","fill hessian part2", "solve linear system" ,"line search energy dicrease","line search locally injective"};
		float elapsed[8];
		for (int i = 0; i < 8; i++)
		{
			cudaEventElapsedTime(&elapsed[i], times[i], times[i + 1]);
			printf("time in %s: %6.2f\n", headers[i], elapsed[i]);
		}*/
	}

	void set_diag(R eps)
	{
		setValue<<<blockNum(3 * nvar), threadsPerBlock>>>(energy_h.pdata, eps, 3 * nvar, 3 * nvar + 1);
	}

	void compute_jacobian()
	{
		gemm(hdl, CUBLAS_OP_T, CUBLAS_OP_N, 3 * nj, 3, nvar, dev_one, Jp.pdata, nvar, z.pdata, nvar, dev_zero, J.pdata, 3 * nj);
	}

	void compute_quad_energy_derivatives();
	void compute_distortion_derivatives();

	void update_fill_derivatives()
	{
		cudaEventRecord(times[0], 0);
		compute_quad_energy_derivatives();
		compute_distortion_derivatives();
	}
	void solve_linear_system()
	{

		energy_h.Factor();
		cudaMemcpyAsync(delta.pdata, energy_g.pdata, 3 * nvar * sizeof(R), cudaMemcpyDeviceToDevice);


		energy_h.Solve(delta.pdata);
		cudaEventRecord(times[6], 0);
		//cudaEventSynchronize(times[6]);			//solve linear system times
	}
	void compute_distortion_energy_derivatives_resp2J();
	void linesearch();
	void show_times();
};

template<class R>
void HarmonicMap<R>::show_times()
{
	cudaEventSynchronize(times[8]);
	const char* headers[] = { "quadric energy", "svd", "jacob diff", "fill gradient", "fill hessian", "solve linear system" ,"line search energy dicrease","line search locally injective" };
	float elapsed[8];

	for (int i = 0; i < 8; i++)
	{
		cudaEventElapsedTime(&elapsed[i], times[i], times[i + 1]);
		printf("time in %s: %6.2f\n", headers[i], elapsed[i]);
	}

}


template<class R>
void HarmonicMap<R>::compute_quad_energy_derivatives()
{
	energy_h.zero_fill();
	geam(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, nvar, dev_two, M.pdata, nvar, dev_zero, NULL, nvar, energy_h.pdata, 3 * nvar);
	geam(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, nvar, dev_two, M.pdata, nvar, dev_zero, NULL, nvar, energy_h.pdata + 3 * nvar * nvar + nvar, 3 * nvar);
	geam(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, nvar, dev_two, M.pdata, nvar, dev_zero, NULL, nvar, energy_h.pdata + 6 * nvar * nvar + 2 * nvar, 3 * nvar);

	if (solver_type == 1)
		set_diag(1e-5);
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, 3, nvar, dev_two, M.pdata, nvar, z.pdata, nvar, dev_zero, energy_g.pdata, nvar);
	geam(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, 3, dev_half, energy_g.pdata, nvar, dev_minus_two, b.pdata, nvar, delta.pdata, nvar); //delta used for a temp variable to compute quadratic energy.
	dot(hdl, nvar * 3, delta.pdata, 1, z.pdata, 1, e);
	cudaMemcpyAsync(q_coef, e, sizeof(R), cudaMemcpyDeviceToDevice);
	axpy(hdl, nvar * 3, dev_minus_two, b.pdata, 1, energy_g.pdata, 1);

	geam(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, 3, dev_minus_one, energy_g.pdata, nvar, dev_zero, NULL, nvar, eq_g.pdata, nvar);
}

template<class R>
void HarmonicMap<R>::compute_distortion_derivatives()
{


	cudaEventRecord(times[1], 0);
	compute_jacobian();
	compute_distortion_energy_derivatives_resp2J();

	/*fill gradient*/
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, 3, 3 * nj, dev_one, Jp.pdata, nvar, dis_energy_gradient_resp2J.pdata, 3 * nj, dev_one, energy_g.pdata, nvar);


	cudaEventRecord(times[4], 0);
	//cudaEventSynchronize(times[4]);			//fill gradient times


	/*fill hessian*/


	float* C1 = temp.pdata;
	float* C2 = temp.pdata + 9 * nh * nvar;
	float* C3 = temp.pdata + 18 * nh * nvar;

	//R** DevArrayQ;	R** DevArrayJp;	R** DevArrayC;	//create array for GEMMBatched
	//const R** HostArrayQ;	const R** HostArrayJp;	const R** HostArrayC;

	/*std::vector<const R*> HostArrayQ, HostArrayJp;
	std::vector<R*> HostArrayC;
	HostArrayQ.resize(3 * nh);
	HostArrayJp.resize(3 * nh);
	HostArrayC.resize(3 * nh);
	for (int i = 0; i < nh; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			HostArrayQ[3 * i + j] = Q.pdata + (3 * i + j) * 27;
			HostArrayJp[3 * i + j] = Jp.pdata + i * ss * 3 * nvar;
			//HostArrayC[3 * i + j] = C.pdata + 9 * nvar * nh * j + i * 9 * nvar;
		}
		HostArrayC[3 * i] = C1 + i * 9 * nvar;
		HostArrayC[3 * i + 1] = C2 + i * 9 * nvar;
		HostArrayC[3 * i + 2] = C3 + i * 9 * nvar;
	}

	DevArrayQ = HostArrayQ;
	DevArrayJp = HostArrayJp;
	DevArrayC = HostArrayC;*/

	gemmBatched(hdl, CUBLAS_OP_N, CUBLAS_OP_T, nvar, 9, 3, dev_const_S.pdata + 1, DevArrayJp.pdata, nvar, DevArrayQ.pdata, 9, dev_const_S.pdata, DevArrayC.pdata, nvar, 3 * nh);
	geam(hdl, CUBLAS_OP_T, CUBLAS_OP_N, 9 * nh, nvar, dev_const_S.pdata + 1, C1, nvar, dev_const_S.pdata, NULL, 9 * nh, C.pdata, 9 * nh);
	geam(hdl, CUBLAS_OP_T, CUBLAS_OP_N, 9 * nh, nvar, dev_const_S.pdata + 1, C2, nvar, dev_const_S.pdata, NULL, 9 * nh, C.pdata + 9 * nvar * nh, 9 * nh);
	geam(hdl, CUBLAS_OP_T, CUBLAS_OP_N, 9 * nh, nvar, dev_const_S.pdata + 1, C3, nvar, dev_const_S.pdata, NULL, 9 * nh, C.pdata + 18 * nvar * nh, 9 * nh);
	convertType << <blockNum(3 * nvar * 3 * nvar), threadsPerBlock >> > (eh_temp.pdata, energy_h.pdata, 3 * nvar * 3 * nvar);

	syrk(hdl, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, 3 * nvar, 9 * nh, dev_const_S.pdata + 2, C.pdata, 
		9 * nh, dev_const_S.pdata + 1, eh_temp.pdata, 3 * nvar); //in most cases, syrk is faster than gemm.
															      //generally, syrk is nearly twice faster as gemm.
	convertType << <blockNum(3 * nvar * 3 * nvar), threadsPerBlock >> > (energy_h.pdata, eh_temp.pdata, 3 * nvar * 3 * nvar);
	//gemm(hdl, CUBLAS_OP_T, CUBLAS_OP_N, 3 * nvar, 3 * nvar, 9 * nh, dev_h_rate, C.pdata, 9 * nh, C.pdata, 9 * nh, dev_one, energy_h.pdata, 3 * nvar);
	if (solver_type == 0)
	{
		fill_up_symmetric_matrix<<<blockNum(9 * nvar * nvar),threadsPerBlock >>>(energy_h.pdata, 3 * nvar, CUBLAS_FILL_MODE_LOWER);
	}
	
	cudaEventRecord(times[5], 0);
	//cudaEventSynchronize(times[5]);			//fill hessian time
}



template<class R>
void HarmonicMap<R>::compute_distortion_energy_derivatives_resp2J()
{
	svd_jacobian_matrix << <blockNum(nj), threadsPerBlock >> > (J.pdata, nj, U.pdata, S.pdata, V.pdata, e, et);


	cudaEventRecord(times[2], 0);
	//cudaEventSynchronize(times[2]);			//svd times

	compute_isometric_energy_gradient << <blockNum(nj), threadsPerBlock >> > (U.pdata, V.pdata, S.pdata, nj, dis_energy_gradient_resp2J.pdata, et);
	compute_isometric_energy_hessian << <blockNum(nh), threadsPerBlock >> > (U.pdata, V.pdata, S.pdata, nh, ss, Q.pdata, et);

	cudaEventRecord(times[3], 0);
	//cudaEventSynchronize(times[3]);			//Jacob derivative times
}



template<class R>
void HarmonicMap<R>::linesearch()
{
	dot(hdl, 3 * nvar, delta.pdata, 1, energy_g.pdata, 1, dot_delta_g);
	nrm2(hdl, 3 * nvar, delta.pdata, 1, delta_norm);

	R* dJ = U.pdata; //re-use the memory of U
	gemm(hdl, CUBLAS_OP_T, CUBLAS_OP_N, nj * 3, 3, nvar, dev_one, Jp.pdata, nvar, delta.pdata, nvar, dev_zero, dJ, 3 * nj);

	dot(hdl, 3 * nvar, eq_g.pdata, 1, delta.pdata, 1, q_coef + 1);
	gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, 3, nvar, dev_one, M.pdata, nvar, delta.pdata, nvar, dev_zero, eq_g.pdata, nvar);
	dot(hdl, 3 * nvar, eq_g.pdata, 1, delta.pdata, 1, q_coef + 2);

	harmonic_linesearch_for_energy_decrease << <1, 1 >> > (J.pdata, nj, dJ, q_coef, dot_delta_g, delta_norm, e, et, ls);

	cudaEventRecord(times[7], 0);

	compute_lips_coef_3(coef_vH.pdata, coef_NH.pdata, ka_norm.pdata, L1.pdata, Ngamma.pdata, Kappa.pdata, z.pdata, delta.pdata, temp1.pdata, temp2.pdata, nl, ne, nvar);

	
	cudaMemcpyAsync(adaptive_sampling_position.pdata + max_number_adaptive_sampling * 3, isp.pdata, 3 * nj * sizeof(real), cudaMemcpyDeviceToDevice);
	cudaMemcpyAsync(adaptive_sampling_father_knot_idx.pdata + max_number_adaptive_sampling, max_idx_array.pdata, nj * sizeof(int), cudaMemcpyDeviceToDevice);
	R* dev_z;
	R* dev_delta;
	if (Proj_z.len)
	{
		gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvnf, 3, nvar, dev_one, ProjMat.pdata, nvnf, z.pdata, nvar, dev_zero, Proj_z.pdata, nvnf);
		dev_z = Proj_z.pdata;
		gemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvnf, 3, nvar, dev_one, ProjMat.pdata, nvnf, delta.pdata, nvar, dev_zero, Proj_delta.pdata, nvnf);
		dev_delta = Proj_delta.pdata;
	}
	else
	{
		dev_z = z.pdata;
		dev_delta = delta.pdata;
	}
	harmonic_linesearch_for_locally_injective3<<<1,1>>>(J.pdata, nj, dJ, lips_constant, dev_sample_radius, dev_sample_radius2, nl, coef_vH.pdata, coef_NH.pdata, ka_norm.pdata, 
		delta_norm, ls, bounds, et, dev_z, dev_delta, adaptive_sampling_position.pdata, dev_ls_step_size,
		flag_list_of_failed.pdata, d_temp_storage.pdata, J_tmp_storage.pdata, H_norm_storage.pdata, num_flip_points.pdata, tmp_s3.pdata, bd.pdata, max_idx_array.pdata, flip_point_idx.pdata, max_number_adaptive_sampling,
		adaptive_sampling_father_knot_idx.pdata,
		cageT_in.pdata, nf, nv, x1.pdata, x2.pdata, x3.pdata, N.pdata, nn.pdata, dx1.pdata, dx2.pdata, dx3.pdata, l1a.pdata, l2a.pdata, l3a.pdata, cNdx1.pdata, cNdx2.pdata, cNdx3.pdata, sig3_low_bound_flag
		);

	cudaEventRecord(times[8], 0);

	//cudaEventSynchronize(times[8]);
	geam(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nvar, 3, dev_one, z.pdata, nvar, ls, delta.pdata, nvar, z.pdata, nvar);
}





static HarmonicMap<real> HM;



