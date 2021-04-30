#pragma once

#include "vec3.h"
#include <vector>
#include <Eigen\eigen>

template<class real>
class green_coords_3d_urago3
{
public:
	green_coords_3d_urago3(const real* x, const int* t, const size_t nv, const size_t nt): nv(nv), nt(nt) 
	{

		x1_array.resize(nt * 3);
		x2_array.resize(nt * 3);
		x3_array.resize(nt * 3);
		N_array.resize(nt * 3);
		s_array.resize(nt);
		dx1_array.resize(nt * 3);
		dx2_array.resize(nt * 3);
		dx3_array.resize(nt * 3);
		l1_array.resize(nt);
		l2_array.resize(nt);
		l3_array.resize(nt);
		cNdx1_array.resize(nt * 3);
		cNdx2_array.resize(nt * 3);
		cNdx3_array.resize(nt * 3);

		t_copy.resize(nt * 3);

		memcpy(t_copy.data(), t, sizeof(int) * nt * 3);

		/*precompute the geometry info */
		for (int i = 0; i < nt; i++)
		{
			int idx1, idx2, idx3;
			idx1 = t[3 * i + 0];
			idx2 = t[3 * i + 1];
			idx3 = t[3 * i + 2];

			copyv3(x + idx1 * 3, x1_array.data() + i * 3);
			copyv3(x + idx2 * 3, x2_array.data() + i * 3);
			copyv3(x + idx3 * 3, x3_array.data() + i * 3);
			
			minusv3(x2_array.data() + i * 3, x3_array.data() + i * 3, dx1_array.data() + i * 3);
			minusv3(x3_array.data() + i * 3, x1_array.data() + i * 3, dx2_array.data() + i * 3);
			minusv3(x1_array.data() + i * 3, x2_array.data() + i * 3, dx3_array.data() + i * 3);

			crossv3(dx2_array.data() + i * 3, dx3_array.data() + i * 3, N_array.data() + i * 3);
			s_array[i] = normv3(N_array.data() + i * 3);
			normalizedv3(N_array.data() + i * 3);

			l1_array[i] = normv3(dx1_array.data() + i * 3);
			l2_array[i] = normv3(dx2_array.data() + i * 3);
			l3_array[i] = normv3(dx3_array.data() + i * 3);

			crossv3(N_array.data() + i * 3, dx1_array.data() + i * 3, cNdx1_array.data() + i * 3);
			crossv3(N_array.data() + i * 3, dx2_array.data() + i * 3, cNdx2_array.data() + i * 3);
			crossv3(N_array.data() + i * 3, dx3_array.data() + i * 3, cNdx3_array.data() + i * 3);
		}
	}

	~green_coords_3d_urago3() {}

	size_t get_nv() { return nv; }
	size_t get_nt() { return nt; }
	size_t get_DOFs() { return nv + nt; }

	void compute_green_coords(const real* p_in, const int np, real* phi_out, real* psi_out);
	void compute_green_coords_gradient(const real* p_in, const int np, real* J_phi_out, real* J_psi_out);
	void compute_green_coords_hessian(const real* p_in, const int np, real* H_phi_out, real* H_psi_out);
private:
	std::vector<int> t_copy;
	std::vector<real> x1_array, x2_array, x3_array, N_array, s_array, dx1_array, dx2_array, dx3_array, l1_array, l2_array, l3_array;
	std::vector<real> cNdx1_array, cNdx2_array, cNdx3_array;
	int nv = 0;
	int nt = 0;
};


template<class real>
void green_coords_3d_urago3<real>::compute_green_coords(const real* p_in, const int np, real* phi_out, real* psi_out)
{

#pragma omp parallel for  
	for (int i = 0; i < np; i++)
		for (int j = 0; j < nt; j++)
		{
			const real* x1 = x1_array.data() + 3 * j;
			const real* x2 = x2_array.data() + 3 * j;
			const real* x3 = x3_array.data() + 3 * j;
			const real* N = N_array.data() + 3 * j;
			const real* s = s_array.data() + j;
			const real* dx1 = dx1_array.data() + 3 * j;
			const real* dx2 = dx2_array.data() + 3 * j;
			const real* dx3 = dx3_array.data() + 3 * j;
			const real* l1 = l1_array.data() + j;
			const real* l2 = l2_array.data() + j;
			const real* l3 = l3_array.data() + j;
			const real* p = p_in + 3 * i;

			real e1[3], e2[3], e3[3];
			minusv3(x1, p, e1);
			minusv3(x2, p, e2);
			minusv3(x3, p, e3);

			real g[3];
			for (int k = 0; k < 3; k++)
			{
				g[k] = (e1[k] + e2[k] + e3[k]) / 3;
			}

			real J1[3], J2[3], J3[3];
			crossv3(e3, e2, J1);
			crossv3(e1, e3, J2);
			crossv3(e2, e1, J3);

			real sR1, sR2, sR3;
			sR1 = normv3(e2) + normv3(e3);
			sR2 = normv3(e3) + normv3(e1);
			sR3 = normv3(e1) + normv3(e2);

			real sc1, sc2, sc3;
			sc1 = 1 / l1[0] * log((sR1 + l1[0]) / (sR1 - l1[0]));
			sc2 = 1 / l2[0] * log((sR2 + l2[0]) / (sR2 - l2[0]));
			sc3 = 1 / l3[0] * log((sR3 + l3[0]) / (sR3 - l3[0]));

			real L[3];
			L[0] = J1[0] * sc1 + J2[0] * sc2 + J3[0] * sc3;
			L[1] = J1[1] * sc1 + J2[1] * sc2 + J3[1] * sc3;
			L[2] = J1[2] * sc1 + J2[2] * sc2 + J3[2] * sc3;

			real M[3];
			M[0] = dx1[0] * sc1 + dx2[0] * sc2 + dx3[0] * sc3;
			M[1] = dx1[1] * sc1 + dx2[1] * sc2 + dx3[1] * sc3;
			M[2] = dx1[2] * sc1 + dx2[2] * sc2 + dx3[2] * sc3;

			real omega;
			omega = -calc_omega(e1, e2, e3);

			real P[3];
			crossv3(N, M, P);
			real temp[3];
			copyv3(N, temp);
			scalv3(omega, temp);
			addv3(P, temp, P);
			scalv3(1 / (4 * M_PI), P);

			// psi & phi
			scalv3(omega, g);
			minusv3(g, L, L);
			psi_out[np * j + i] = dotv3(L, N) / (4 * M_PI);

#pragma omp atomic
				phi_out[t_copy[j * 3 + 0] * np + i] += dotv3(J1, P) / s[0];
#pragma omp atomic
				phi_out[t_copy[j * 3 + 1] * np + i] += dotv3(J2, P) / s[0];
#pragma omp atomic
				phi_out[t_copy[j * 3 + 2] * np + i] += dotv3(J3, P) / s[0];
			

		}
}

template<class real>
void green_coords_3d_urago3<real>::compute_green_coords_gradient(const real* p_in, const int np, real* J_phi_out, real* J_psi_out)
{
#pragma omp parallel for  
	for (int i = 0; i < np; i++)
		for (int j = 0; j < nt; j++)
		{
			const real* x1 = x1_array.data() + 3 * j;
			const real* x2 = x2_array.data() + 3 * j;
			const real* x3 = x3_array.data() + 3 * j;
			const real* N = N_array.data() + 3 * j;
			const real* s = s_array.data() + j;
			const real* dx1 = dx1_array.data() + 3 * j;
			const real* dx2 = dx2_array.data() + 3 * j;
			const real* dx3 = dx3_array.data() + 3 * j;
			const real* l1 = l1_array.data() + j;
			const real* l2 = l2_array.data() + j;
			const real* l3 = l3_array.data() + j;
			const real* p = p_in + 3 * i;

			real e1[3], e2[3], e3[3];
			minusv3(x1, p, e1);
			minusv3(x2, p, e2);
			minusv3(x3, p, e3);

			real J1[3], J2[3], J3[3];
			crossv3(e3, e2, J1);
			crossv3(e1, e3, J2);
			crossv3(e2, e1, J3);

			real sR1, sR2, sR3;
			sR1 = normv3(e2) + normv3(e3);
			sR2 = normv3(e3) + normv3(e1);
			sR3 = normv3(e1) + normv3(e2);

			real sc1, sc2, sc3;
			sc1 = 1 / l1[0] * log((sR1 + l1[0]) / (sR1 - l1[0]));
			sc2 = 1 / l2[0] * log((sR2 + l2[0]) / (sR2 - l2[0]));
			sc3 = 1 / l3[0] * log((sR3 + l3[0]) / (sR3 - l3[0]));

			real M[3];
			M[0] = dx1[0] * sc1 + dx2[0] * sc2 + dx3[0] * sc3;
			M[1] = dx1[1] * sc1 + dx2[1] * sc2 + dx3[1] * sc3;
			M[2] = dx1[2] * sc1 + dx2[2] * sc2 + dx3[2] * sc3;

			real omega;
			omega = -calc_omega(e1, e2, e3);

			real P[3];
			crossv3(N, M, P);
			real temp[3];
			copyv3(N, temp);
			scalv3(omega, temp);
			addv3(P, temp, P);
			scalv3(1 / (4 * M_PI), P);

			copyv3(P, J_psi_out + j * np * 3 + i * 3);
			scalv3(real(-1), J_psi_out + j * np * 3 + i * 3);

			real J_phi[9];	

			crossv3(P, dx1, J_phi);
			crossv3(P, dx2, J_phi + 3);
			crossv3(P, dx3, J_phi + 6);

			scalv3(real(1) / s[0], J_phi);
			scalv3(real(1) / s[0], J_phi + 3);
			scalv3(real(1) / s[0], J_phi + 6);
			for (int k = 0; k < 3; k++)
			{
#pragma omp atomic
				J_phi_out[t_copy[j * 3 + 0] * 3 * np + 3 * i + k] += J_phi[0 + k];
#pragma omp atomic
				J_phi_out[t_copy[j * 3 + 1] * 3 * np + 3 * i + k] += J_phi[3 + k];
#pragma omp atomic
				J_phi_out[t_copy[j * 3 + 2] * 3 * np + 3 * i + k] += J_phi[6 + k];
			}
		}

}


template<class real>
void green_coords_3d_urago3<real>::compute_green_coords_hessian(const real* p_in, const int np, real* H_phi_out, real* H_psi_out)
{
#pragma omp parallel for  
	for (int i = 0; i < np; i++)
		for (int j = 0; j < nt; j++)
		{
			const real* x1 = x1_array.data() + 3 * j;
			const real* x2 = x2_array.data() + 3 * j;
			const real* x3 = x3_array.data() + 3 * j;
			const real* N = N_array.data() + 3 * j;
			const real* s = s_array.data() + j;
			const real* dx1 = dx1_array.data() + 3 * j;
			const real* dx2 = dx2_array.data() + 3 * j;
			const real* dx3 = dx3_array.data() + 3 * j;
			const real* l1 = l1_array.data() + j;
			const real* l2 = l2_array.data() + j;
			const real* l3 = l3_array.data() + j;
			const real* cNdx1 = cNdx1_array.data() + 3 * j;
			const real* cNdx2 = cNdx2_array.data() + 3 * j;
			const real* cNdx3 = cNdx3_array.data() + 3 * j;
			const real* p = p_in + 3 * i;


			real e1[3], e2[3], e3[3];
			minusv3(x1, p, e1);
			minusv3(x2, p, e2);
			minusv3(x3, p, e3);

			real temp[9];
			real* J1 = temp;
			real* J2 = temp + 3;
			real* J3 = temp + 6;
			crossv3(e3, e2, J1);
			crossv3(e1, e3, J2);
			crossv3(e2, e1, J3);

			real sR1, sR2, sR3;
			sR1 = normv3(e2) + normv3(e3);
			sR2 = normv3(e3) + normv3(e1);
			sR3 = normv3(e1) + normv3(e2);

			real oR1, oR2, oR3;
			oR1 = 1 / normv3(e2) + 1 / normv3(e3);
			oR2 = 1 / normv3(e3) + 1 / normv3(e1);
			oR3 = 1 / normv3(e1) + 1 / normv3(e2);

			normalizedv3(e1);
			normalizedv3(e2);
			normalizedv3(e3);

			real dsc1, dsc2, dsc3;
			dsc1 = 2 / ((sR1 + l1[0]) * (sR1 - l1[0]));
			dsc2 = 2 / ((sR2 + l2[0]) * (sR2 - l2[0]));
			dsc3 = 2 / ((sR3 + l3[0]) * (sR3 - l3[0]));

			real Et[9];
			addv3(e2, e3, Et);
			scalv3(dsc1, Et);
			addv3(e3, e1, Et + 3);
			scalv3(dsc2, Et + 3);
			addv3(e1, e2, Et + 6);
			scalv3(dsc3, Et + 6);

			real dscor[3] = { dsc1 * oR1, dsc2 * oR2, dsc3 * oR3 };

			real grad_omega_x, grad_omega_y, grad_omega_z;
			grad_omega_x = J1[0] * dscor[0] + J2[0] * dscor[1] + J3[0] * dscor[2];
			grad_omega_y = J1[1] * dscor[0] + J2[1] * dscor[1] + J3[1] * dscor[2];
			grad_omega_z = J1[2] * dscor[0] + J2[2] * dscor[1] + J3[2] * dscor[2];

			real* J_P = temp;
			J_P[0] = (Et[0] * cNdx1[0] + Et[3] * cNdx2[0] + Et[6] * cNdx3[0] + grad_omega_x * N[0]) / (4 * M_PI);
			J_P[1] = (Et[1] * cNdx1[0] + Et[4] * cNdx2[0] + Et[7] * cNdx3[0] + grad_omega_y * N[0]) / (4 * M_PI);
			J_P[2] = (Et[2] * cNdx1[0] + Et[5] * cNdx2[0] + Et[8] * cNdx3[0] + grad_omega_z * N[0]) / (4 * M_PI);

			J_P[3] = (Et[0] * cNdx1[1] + Et[3] * cNdx2[1] + Et[6] * cNdx3[1] + grad_omega_x * N[1]) / (4 * M_PI);
			J_P[4] = (Et[1] * cNdx1[1] + Et[4] * cNdx2[1] + Et[7] * cNdx3[1] + grad_omega_y * N[1]) / (4 * M_PI);
			J_P[5] = (Et[2] * cNdx1[1] + Et[5] * cNdx2[1] + Et[8] * cNdx3[1] + grad_omega_z * N[1]) / (4 * M_PI);

			J_P[6] = (Et[0] * cNdx1[2] + Et[3] * cNdx2[2] + Et[6] * cNdx3[2] + grad_omega_x * N[2]) / (4 * M_PI);
			J_P[7] = (Et[1] * cNdx1[2] + Et[4] * cNdx2[2] + Et[7] * cNdx3[2] + grad_omega_y * N[2]) / (4 * M_PI);
			J_P[8] = (Et[2] * cNdx1[2] + Et[5] * cNdx2[2] + Et[8] * cNdx3[2] + grad_omega_z * N[2]) / (4 * M_PI);

			H_psi_out[j * np * 5 + i * 5 + 0] = -J_P[0];
			H_psi_out[j * np * 5 + i * 5 + 1] = -J_P[1];
			H_psi_out[j * np * 5 + i * 5 + 2] = -J_P[2];
			H_psi_out[j * np * 5 + i * 5 + 3] = -J_P[4];
			H_psi_out[j * np * 5 + i * 5 + 4] = -J_P[5];

			real H_phi[15];

			real* temp2 = dscor;
			crossv3(dx1, J_P, temp2);
			H_phi[0] = (-temp2[0]) / s[0];
			crossv3(dx1, J_P + 3, temp2);
			H_phi[1] = (-temp2[0]) / s[0];
			H_phi[3] = (-temp2[1]) / s[0];
			crossv3(dx1, J_P + 6, temp2);
			H_phi[2] = (-temp2[0]) / s[0];
			H_phi[4] = (-temp2[1]) / s[0];

			crossv3(dx2, J_P, temp2);
			H_phi[5] = (-temp2[0]) / s[0];
			crossv3(dx2, J_P + 3, temp2);
			H_phi[6] = (-temp2[0]) / s[0];
			H_phi[8] = (-temp2[1]) / s[0];
			crossv3(dx2, J_P + 6, temp2);
			H_phi[7] = (-temp2[0]) / s[0];
			H_phi[9] = (-temp2[1]) / s[0];

			crossv3(dx3, J_P, temp2);
			H_phi[10] = (-temp2[0]) / s[0];
			crossv3(dx3, J_P + 3, temp2);
			H_phi[11] = (-temp2[0]) / s[0];
			H_phi[13] = (-temp2[1]) / s[0];
			crossv3(dx3, J_P + 6, temp2);
			H_phi[12] = (-temp2[0]) / s[0];
			H_phi[14] = (-temp2[1]) / s[0];
			for (int k = 0; k < 5; k++)
			{
#pragma omp atomic
				H_phi_out[t_copy[3 * j + 0] * 5 * np + 5 * i + k] += H_phi[k];
#pragma omp atomic
				H_phi_out[t_copy[3 * j + 1] * 5 * np + 5 * i + k] += H_phi[5 + k];
#pragma omp atomic
				H_phi_out[t_copy[3 * j + 2] * 5 * np + 5 * i + k] += H_phi[10 + k];			
			}
		}
}


