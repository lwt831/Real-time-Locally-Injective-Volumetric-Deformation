#include <mex.h>
#include <string>
#include <vector>
#include "vec3.h"


template<class real>
void lipschitz_kernel_matrix_kappa(const real* x, const int nv, const int* t, const int* edge_info, const int ne,
	const real* N, const real* area2, real* kappa)
{
#pragma omp parallel for  
	for (int i = 0; i < ne; i++)
	{
		int v1 = edge_info[i * 4 + 0];
		int v2 = edge_info[i * 4 + 1];
		int f1 = edge_info[i * 4 + 2];
		int f2 = edge_info[i * 4 + 3];
		real* NN = NULL;
		real* NXdv = NULL;
		real a;
		real temp[3];  // dvi
		real temp2[3]; //N2 cross pdvi

		/*face 1*/
		NN = kappa + 3 * ne * (nv + f1) + 3 * i; //NN = N1
		copyv3(N + f1 * 3, NN);

		a = real(1) / area2[f1];

		minusv3(x + 3 * (t[3 * f1 + 1]), x + 3 * (t[3 * f1 + 2]), temp); // dv1
		scalv3(a, temp);
		NXdv = kappa + 3 * ne * (t[3 * f1]) + 3 * i;
		crossv3(temp, NN, NXdv);  //N1 cross dv1

		minusv3(x + 3 * (t[3 * f1 + 2]), x + 3 * (t[3 * f1 + 0]), temp); //dv2
		scalv3(a, temp);
		NXdv = kappa + 3 * ne * (t[3 * f1 + 1]) + 3 * i;
		crossv3(temp, NN, NXdv);  //N1 cross dv2

		minusv3(x + 3 * (t[3 * f1 + 0]), x + 3 * (t[3 * f1 + 1]), temp); //dv3
		scalv3(a, temp);
		NXdv = kappa + 3 * ne * (t[3 * f1 + 2]) + 3 * i;
		crossv3(temp, NN, NXdv);  //N1 cross dv3


		/*face 2*/
		NN = kappa + 3 * ne * (nv + f2) + 3 * i; //N = N2
		copyv3(N + f2 * 3, NN);
		scalv3(real(-1), NN);

		a = real(1) / area2[f2];

		minusv3(x + 3 * (t[3 * f2 + 1]), x + 3 * (t[3 * f2 + 2]), temp); // pdv1
		scalv3(a, temp);
		NXdv = kappa + 3 * ne * (t[3 * f2]) + 3 * i;
		crossv3(temp, NN, temp2);
		addv3(NXdv, temp2, NXdv);

		minusv3(x + 3 * (t[3 * f2 + 2]), x + 3 * (t[3 * f2 + 0]), temp); // pdv2
		scalv3(a, temp);
		NXdv = kappa + 3 * ne * (t[3 * f2 + 1]) + 3 * i;
		crossv3(temp, NN, temp2);
		addv3(NXdv, temp2, NXdv);

		minusv3(x + 3 * (t[3 * f2 + 0]), x + 3 * (t[3 * f2 + 1]), temp); // pdv3
		scalv3(a, temp);
		NXdv = kappa + 3 * ne * (t[3 * f2 + 2]) + 3 * i;
		crossv3(temp, NN, temp2);
		addv3(NXdv, temp2, NXdv);
	}

}

/*this function fails when the segment between p1 and p2 passes through the sphere*/
template<class real>
 real find_minimum_sum_distances_between_two_points_to_sphere(const real* p1, const real* p2, const real* o, const real r) // minimize_p0 (|p1 - p0| + |p2 - p0|) s.t |p0 - o| = r
{
	real e1[3];
	real e2[3];
	real Nx[3] = { 0, 0, 0 };
	minusv3(p1, o, e1);
	minusv3(p2, o, e2);
	scalv3(real(1) / r, e1);
	scalv3(real(1) / r, e2); //normalize sphere to unit shpere
	real a = dotv3(e1, e1);
	real b = dotv3(e1, e2);
	real c = dotv3(e2, e2);
	real A = 4 * c * (a * c - b * b);
	real B = -4 * (a * c - b * b);
	real C = (a + 2 * b + c - 4 * a * c);
	real D = 2 * (a - b);
	real E = a - 1;
	if (A < 1e-6) //in this case, p1, p2 and o are collinear
	{
		copyv3(e1, Nx);
		scalv3(sqrt(a), Nx);
	}

	//solve quartic equation Ax^4 + Bx^3 + Cx^2 + Dx + E = 0
	real pa1 = (8 * A * C - 3 * B * B) / (8 * A * A);
	real pa2 = (B * B * B - 4 * A * B * C + 8 * A * A * D) / (8 * A * A * A);
	real delta0 = C * C - 3 * B * D + 12 * A * E;
	real delta1 = 2 * C * C * C - 9 * B * C * D + 27 * B * B * E + 27 * A * D * D - 72 * A * C * E;
	real theta = acos(delta1 / (2 * pow(delta0, 1.5)));
	real pa3 = 1.0 / 2 * sqrt(-2.0 / 3 * pa1 + 2.0 / (3 * A) * sqrt(delta0) * cos(theta / 3));
	real k1 = -4 * pa3 * pa3 - 2 * pa1 + pa2 / pa3;
	k1 = k1 > 0 ? k1 : 0;
	real k2 = -4 * pa3 * pa3 - 2 * pa1 - pa2 / pa3;
	k2 = k2 > 0 ? k2 : 0; // avoid numerical fault
	real x, y;
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

 template<class real>
 void Lipschitz_Kernel_Matrix_Ngamma(const real* x, const int nv, const int* t, const int* Half_edge_Info, const int ne, const real* p_in, const int np, real* eps, real* Ngamma)
 {
#pragma omp parallel for  
	 for (int i = 0; i < np; i++)
		 for(int j = 0; j < ne; j++)
		{
			 int v1 = Half_edge_Info[j * 4];
			 int v2 = Half_edge_Info[j * 4 + 1];
			 const real* p = p_in + i * 3;

			 real r = sqrt(3.0) / 3 * eps[0];

			 real R1 = find_minimum_sum_distances_between_two_points_to_sphere(x + v1 * 3, x + v2 * 3, p, r);
			 real e1[3];
			 real e2[3];
			 minusv3(x + v1 * 3, p, e1);
			 minusv3(x + v2 * 3, p, e2);
			 real d[3];
			 minusv3(x + v2 * 3, x + v1 * 3, d);
			 real nd = normv3(d);
			 real l1 = normv3(e1);
			 real l2 = normv3(e2);
			 real C = real(1) / 2 / M_PI / (R1 * R1 - nd * nd);
			 Ngamma[j * np + i] = C * sqrt((4 + 10 * R1 * R1 * nd * nd / (l1 - r) / (l1 - r) / (l2 - r) / (l2 - r)));
		}
 }

#ifdef _DEBUG

#include "MatlabEngine.h"
#include <ctime>
#include<iostream>

int main()
{
	getMatEngine().connect("");
	const mxArray* x_mat, * t_mat;
	x_mat = getMatEngine().getVariable("cx");
	t_mat = getMatEngine().getVariable("ct");
	const mxArray* edge_info_mat = getMatEngine().getVariable("edge_info");
	size_t nv = mxGetN(x_mat);
	size_t nt = mxGetN(t_mat);

	const double* x = mxGetDoubles(x_mat);
	const int* t = mxGetInt32s(t_mat);
	const int* edge_info = mxGetInt32s(edge_info_mat);

	const size_t ne = nv + nt - 2;

	std::vector<double> N;
	std::vector<double> area2;
	N.resize(3 * nt);
	area2.resize(nt);
	for (int i = 0; i < nt; i++)
	{
		int idx1, idx2, idx3;
		idx1 = t[3 * i + 0];
		idx2 = t[3 * i + 1];
		idx3 = t[3 * i + 2];

		double x1[3], x2[3], x3[3];
		double dx1[3], dx2[3], dx3[3];

		copyv3(x + idx1 * 3, x1);
		copyv3(x + idx2 * 3, x2);
		copyv3(x + idx3 * 3, x3);

		minusv3(x2, x3, dx1);
		minusv3(x3, x1, dx2);
		minusv3(x1, x2, dx3);

		crossv3(dx2, dx3, N.data() + i * 3);
		area2[i] = normv3(N.data() + i * 3);
		normalizedv3(N.data() + i * 3);
	}
	mxArray* kappa_mat = mxCreateDoubleMatrix(ne * 3, nv + nt, mxREAL);
	double* kappa = mxGetDoubles(kappa_mat);
	std::clock_t start, end;
	double endtime;

	start = clock();
	lipschitz_kernel_matrix_kappa(x, nv, t, edge_info, ne,
		N.data(), area2.data(), kappa);
	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "time of lipchitz_kernel_matrix_kappa:" << endtime << std::endl;

	getMatEngine().putVariable("Kappa_out", kappa_mat);

	const mxArray* p_in_mat = getMatEngine().getVariable("p_in");
	double* p_in = mxGetDoubles(p_in_mat);
	size_t np = mxGetN(p_in_mat);

	const mxArray* eps_mat = getMatEngine().getVariable("eps");
	double* eps = mxGetDoubles(eps_mat);

	mxArray* Ngamma_mat = mxCreateDoubleMatrix(np, ne, mxREAL);
	double* Ngamma = mxGetDoubles(Ngamma_mat);
	//mxArray* plhs[2] = mxCreateDoubleMatrix(np, ne, mxREAL);
	start = clock();
	Lipschitz_Kernel_Matrix_Ngamma(x, nv, t, edge_info, ne, p_in, np, eps, Ngamma);
	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "time of lipchitz_kernel_matrix_Ngamma:" << endtime << std::endl;

	getMatEngine().putVariable("Ngamma_mat_out", Ngamma_mat);
}

#else

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// Gamma not be computed , return a zero-matrix( gamma be used to second order estimate)
	if (nrhs < 1)
		mexErrMsgTxt("Invalid input: not enough input, [Kappa, Ngamma, Gamma] = lip_kernel_matrix_c(x, t, edge_info, p_in, eps)");
	const double* x = mxGetDoubles(prhs[0]);
	const int* t = mxGetInt32s(prhs[1]);
	const int* edge_info = mxGetInt32s(prhs[2]);

	size_t nv = mxGetN(prhs[0]);
	size_t nt = mxGetN(prhs[1]);

	const size_t ne = nv + nt - 2;

	std::vector<double> N;
	std::vector<double> area2;
	N.resize(3 * nt);
	area2.resize(nt);
	for (int i = 0; i < nt; i++)
	{
		int idx1, idx2, idx3;
		idx1 = t[3 * i + 0];
		idx2 = t[3 * i + 1];
		idx3 = t[3 * i + 2];

		double x1[3], x2[3], x3[3];
		double dx1[3], dx2[3], dx3[3];

		copyv3(x + idx1 * 3, x1);
		copyv3(x + idx2 * 3, x2);
		copyv3(x + idx3 * 3, x3);

		minusv3(x2, x3, dx1);
		minusv3(x3, x1, dx2);
		minusv3(x1, x2, dx3);

		crossv3(dx2, dx3, N.data() + i * 3);
		area2[i] = normv3(N.data() + i * 3);
		normalizedv3(N.data() + i * 3);
	}

	plhs[0] = mxCreateDoubleMatrix(ne * 3, nv + nt, mxREAL);
	double* kappa = mxGetDoubles(plhs[0]);

	lipschitz_kernel_matrix_kappa(x, nv, t, edge_info, ne,
		N.data(), area2.data(), kappa);
	if (nrhs < 4)
		return;

	double* p_in = mxGetDoubles(prhs[3]);
	size_t np = mxGetN(prhs[3]);
	double* eps = mxGetDoubles(prhs[4]);
	plhs[1] = mxCreateDoubleMatrix(np, ne, mxREAL);
	double* Ngamma = mxGetDoubles(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(np, ne, mxREAL);
	Lipschitz_Kernel_Matrix_Ngamma(x, nv, t, edge_info, ne, p_in, np, eps, Ngamma);
}
#endif