#pragma once
#include "cuVec3.cuh"
#include "CublasTempFunctions.cuh"
#include "utils.cuh"

using real = double;
extern real* dev_one;
extern real* dev_zero;
extern real* dev_minus_one;
extern real* dev_two;
extern real* dev_minus_two;
extern real* dev_half;


template<class R>
__global__ void Lipschitz_Kernel_Matrix_Kappa(const R* cageX, const int nv, const int* cageT, const int* Half_edge_Info, const int ne, const R* N, const R* area2, R* Kappa)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < ne)
	{
		int vid1 = Half_edge_Info[tid * 4];
		int vid2 = Half_edge_Info[tid * 4 + 1];
		int fid1 = Half_edge_Info[tid * 4 + 2];
		int fid2 = Half_edge_Info[tid * 4 + 3];
		R* NN = NULL;
		R* NXdv = NULL;
		R a;
		R temp[3];  // dvi
		R temp2[3]; //N2 cross pdvi

		/*face 1*/
		NN = Kappa + 3 * ne * (nv + fid1) + 3 * tid; //NN = N1
		copyv3(N + fid1 * 3, NN);

		a = R(1) / area2[fid1];

		minusv3(cageX + 3 * (cageT[3 * fid1 + 1]), cageX + 3 * (cageT[3 * fid1 + 2]), temp); // dv1
		scalv3(a, temp);
		NXdv = Kappa + 3 * ne * (cageT[3 * fid1]) + 3 * tid;
		crossv3(temp, NN, NXdv);  //N1 cross dv1

		minusv3(cageX + 3 * (cageT[3 * fid1 + 2]), cageX + 3 * (cageT[3 * fid1 + 0]), temp); //dv2
		scalv3(a, temp);
		NXdv = Kappa + 3 * ne * (cageT[3 * fid1 + 1]) + 3 * tid;
		crossv3(temp, NN, NXdv);  //N1 cross dv2

		minusv3(cageX + 3 * (cageT[3 * fid1 + 0]), cageX + 3 * (cageT[3 * fid1 + 1]), temp); //dv3
		scalv3(a, temp);
		NXdv = Kappa + 3 * ne * (cageT[3 * fid1 + 2]) + 3 * tid;
		crossv3(temp, NN, NXdv);  //N1 cross dv3


		/*face 2*/
		NN = Kappa + 3 * ne * (nv + fid2) + 3 * tid; //N = N2
		copyv3(N + fid2 * 3, NN);
		scalv3(R(-1), NN);

		a = R(1) / area2[fid2];

		minusv3(cageX + 3 * (cageT[3 * fid2 + 1]), cageX + 3 * (cageT[3 * fid2 + 2]), temp); // pdv1
		scalv3(a, temp);
		NXdv = Kappa + 3 * ne * (cageT[3 * fid2]) + 3 * tid;
		crossv3(temp, NN, temp2);
		addv3(NXdv, temp2, NXdv);

		minusv3(cageX + 3 * (cageT[3 * fid2 + 2]), cageX + 3 * (cageT[3 * fid2 + 0]), temp); // pdv2
		scalv3(a, temp);
		NXdv = Kappa + 3 * ne * (cageT[3 * fid2 + 1]) + 3 * tid;
		crossv3(temp, NN, temp2);
		addv3(NXdv, temp2, NXdv);

		minusv3(cageX + 3 * (cageT[3 * fid2 + 0]), cageX + 3 * (cageT[3 * fid2 + 1]), temp); // pdv3
		scalv3(a, temp);
		NXdv = Kappa + 3 * ne * (cageT[3 * fid2 + 2]) + 3 * tid;
		crossv3(temp, NN, temp2);
		addv3(NXdv, temp2, NXdv);
	}
}



/*this function fails when the segment between p1 and p2 passes through the sphere*/
template<class R>
__device__ R find_minimum_sum_distances_between_two_points_to_sphere(const R* p1, const R* p2, const R* o, const R r) // minimize_p0 (|p1 - p0| + |p2 - p0|) s.t |p0 - o| = r
{
	R e1[3];
	R e2[3];
	R Nx[3] = { 0, 0, 0 };
	minusv3(p1, o, e1);
	minusv3(p2, o, e2);
	scalv3(R(1) / r, e1);
	scalv3(R(1) / r, e2); //normalize sphere to unit shpere
	R a = dotv3(e1, e1);
	R b = dotv3(e1, e2);
	R c = dotv3(e2, e2);
	R A = 4 * c * (a * c - b * b);
	R B = -4 * (a * c - b * b);
	R C = (a + 2 * b + c - 4 * a * c);
	R D = 2 * (a - b);
	R E = a - 1;
	if (A < 1e-6) //in this case, p1, p2 and o are collinear
	{
		copyv3(e1, Nx);
		scalv3(sqrt(a), Nx);
	}

	//solve quartic equation Ax^4 + Bx^3 + Cx^2 + Dx + E = 0
	R pa1 = (8 * A * C - 3 * B * B) / (8 * A * A);
	R pa2 = (B * B * B - 4 * A * B * C + 8 * A * A * D) / (8 * A * A * A);
	R delta0 = C * C - 3 * B * D + 12 * A * E;
	R delta1 = 2 * C * C * C - 9 * B * C * D + 27 * B * B * E + 27 * A * D * D - 72 * A * C * E;
	R theta = acos(delta1 / (2 * pow(delta0, 1.5)));
	R pa3 = 1.0 / 2 * sqrt(-2.0 / 3 * pa1 + 2.0 / (3 * A) * sqrt(delta0) * cos(theta / 3));
	R k1 = -4 * pa3 * pa3 - 2 * pa1 + pa2 / pa3;
	k1 = k1 > 0 ? k1 : 0;
	R k2 = -4 * pa3 * pa3 - 2 * pa1 - pa2 / pa3;
	k2 = k2 > 0 ? k2 : 0; // avoid numerical fault
	R x, y;
	y = -B / (4 * A) - pa3 + 1.0 / 2 * sqrt(k1);
	x = (-2 * c * y * y + y + 1) / (2 * b * y + 1);

	if (y > 0 && y < 1 && x > 0 && x < 1)
	{
		for (int j = 0; j < 3; j++)
		{
			Nx[j] = x * e1[j] + y * e2[j];
		}
	}
	else
	{
		y = -B / (4 * A) - pa3 - 1.0 / 2 * sqrt(k1);
		x = (-2 * c * y * y + y + 1) / (2 * b * y + 1);
		if (y > 0 && y < 1 && x > 0 && x < 1)
			for (int j = 0; j < 3; j++)
			{
				Nx[j] = x * e1[j] + y * e2[j];
			}
		else
		{
			y = -B / (4 * A) + pa3 + 1.0 / 2 * sqrt(k2);
			x = (-2 * c * y * y + y + 1) / (2 * b * y + 1);
			if (y > 0 && y < 1 && x > 0 && x < 1)
				for (int j = 0; j < 3; j++)
				{
					Nx[j] = x * e1[j] + y * e2[j];
				}
			else
			{
				y = -B / (4 * A) + pa3 - 1.0 / 2 * sqrt(k2);
				x = (-2 * c * y * y + y + 1) / (2 * b * y + 1);
				if (y > 0 && y < 1 && x > 0 && x < 1)
					for (int j = 0; j < 3; j++)
					{
						Nx[j] = x * e1[j] + y * e2[j];
					}
			}
		}
	}
	minusv3(e1, Nx, e1);
	minusv3(e2, Nx, e2);
	return r * (normv3(e1) + normv3(e2));
}

#ifdef _DEBUG
#undef printf
#endif // DEBUG


template<class R>
__global__ void Lipschitz_Kernel_Matrix_Ngamma(const R* cageX, const int nv, const int* cageT, const int* Half_edge_Info, const int ne, const R* p_in, const int np, R* eps, R* Ngamma)
{
	int n = ceil(R(ne) / threadsPerBlock);
	int pid = blockIdx.x;

	if (pid < np)
	{
		for (int i = 0; i < n; i++)
		{
			int eid = threadIdx.x + i * threadsPerBlock;
			if (eid < ne)
			{
				int vid1 = Half_edge_Info[eid * 4];
				int vid2 = Half_edge_Info[eid * 4 + 1];
				const R* p = p_in + pid * 3;

				const R r = sqrt(3.0) / 3 * eps[0];

				R R1 = find_minimum_sum_distances_between_two_points_to_sphere(cageX + vid1 * 3, cageX + vid2 * 3, p, r);

				R e1[3];
				R e2[3];
				minusv3(cageX + vid1 * 3, p, e1);
				minusv3(cageX + vid2 * 3, p, e2);
				R d[3];
				minusv3(cageX + vid2 * 3, cageX + vid1 * 3, d);
				R nd = normv3(d);
				R l1 = normv3(e1);
				R l2 = normv3(e2);
				R C = R(1) / 2 / CUDART_PI / (R1 * R1 - nd * nd);
				Ngamma[eid * np + pid] = C * sqrt((4 + 10 * R1 * R1 * nd * nd / (l1 - r) / (l1 - r) / (l2 - r) / (l2 - r))); // |Ngamma| = sqrt(|Ngamma1|^2 + |Ngamma2|^2)
			}
		}
	}
}



template<class R>
__global__ void compute_vertex_Hessian_coef_array(R* coef, const R* temp1, const R* temp2, const int np)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < np)
	{
		R* coef_ = coef + tid * 3;
		coef_[0] = 0;
		coef_[1] = 0;
		coef_[2] = 0;
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 3; j++)
			{
				coef_[0] += temp1[j * 5 * np + tid * 5 + i] * temp1[j * 5 * np + tid * 5 + i];
				coef_[1] -= 2 * temp1[j * 5 * np + tid * 5 + i] * temp2[j * 5 * np + tid * 5 + i];
				coef_[2] += temp2[j * 5 * np + tid * 5 + i] * temp2[j * 5 * np + tid * 5 + i];
			}
	}
}

template<class R>
__global__ void compute_kappa_coef(R* temp1, R* temp2, R* k_norm, const int ne)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < ne)
	{
		R* k_norm_ = k_norm + tid * 2;
		k_norm_[0] = 0;
		k_norm_[1] = 0;

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				k_norm_[0] += temp1[j * 3 * ne + 3 * tid + i] * temp1[j * 3 * ne + 3 * tid + i];
				k_norm_[1] += temp2[j * 3 * ne + 3 * tid + i] * temp2[j * 3 * ne + 3 * tid + i];
			}
		k_norm_[0] = sqrt(k_norm_[0]);
		k_norm_[1] = sqrt(k_norm_[1]);

	}
}



template<class R>
__global__ void compute_lips_global3(const R* coef_vH, const R* coef_NH, const R t, const R r, R* lips, const int np)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < np)
	{
		R lips_;
		lips_ = sqrt(coef_vH[3 * tid] + t * coef_vH[3 * tid + 1] + t * t * coef_vH[3 * tid + 2]);
		if (coef_NH)
			lips_ = lips_ + r * (coef_NH[tid * 2] + t * coef_NH[tid * 2 + 1]);

		atomicMaxPositive(lips, lips_);
	}
}


