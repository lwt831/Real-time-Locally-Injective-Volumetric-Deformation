#pragma once
#include "math_constants.h"
#include"cuVec3.cuh"


template<class R>
__device__ void green_coords_3d_urago3(const R* x1, const R* x2, const R* x3, const R* N, const R* nn, const R* dx1, const R* dx2, const R* dx3, const R* l1a, const R* l2a, const R* l3a, const R* p,
	R* phi, R* psi)
{
	R e1[3], e2[3], e3[3];
	minusv3(x1, p, e1);
	minusv3(x2, p, e2);
	minusv3(x3, p, e3);

	R g[3];
	for (int i = 0; i < 3; i++)
	{
		g[i] = (e1[i] + e2[i] + e3[i]) / 3;
	}

	R J1[3], J2[3], J3[3];
	crossv3(e3, e2, J1);
	crossv3(e1, e3, J2);
	crossv3(e2, e1, J3);

	R sR1, sR2, sR3;
	sR1 = normv3(e2) + normv3(e3);
	sR2 = normv3(e3) + normv3(e1);
	sR3 = normv3(e1) + normv3(e2);

	R sc1, sc2, sc3;
	sc1 = 1 / l1a[0] * log((sR1 + l1a[0]) / (sR1 - l1a[0]));
	sc2 = 1 / l2a[0] * log((sR2 + l2a[0]) / (sR2 - l2a[0]));
	sc3 = 1 / l3a[0] * log((sR3 + l3a[0]) / (sR3 - l3a[0]));

	R L[3];
	L[0] = J1[0] * sc1 + J2[0] * sc2 + J3[0] * sc3;
	L[1] = J1[1] * sc1 + J2[1] * sc2 + J3[1] * sc3;
	L[2] = J1[2] * sc1 + J2[2] * sc2 + J3[2] * sc3;

	R M[3];
	M[0] = dx1[0] * sc1 + dx2[0] * sc2 + dx3[0] * sc3;
	M[1] = dx1[1] * sc1 + dx2[1] * sc2 + dx3[1] * sc3;
	M[2] = dx1[2] * sc1 + dx2[2] * sc2 + dx3[2] * sc3;

	R omega;
	omega = -calc_omega(e1, e2, e3);

	R P[3];
	crossv3(N, M, P);
	R temp[3];
	copyv3(N, temp);
	scalv3(omega, temp);
	addv3(P, temp, P);
	scalv3(1 / (4 * CUDART_PI), P);

	// psi & phi
	scalv3(omega, g);
	minusv3(g, L, L);
	psi[0] = dotv3(L, N) / (4 * CUDART_PI);

	phi[0] = dotv3(J1, P) / nn[0];
	phi[1] = dotv3(J2, P) / nn[0];
	phi[2] = dotv3(J3, P) / nn[0];
}


template<class R>
__device__ void green_coords_3d_urago3_gradient(const R* x1, const R* x2, const R* x3, const R* N, const R* nn, const R* dx1, const R* dx2, const R* dx3, const R* l1a, const R* l2a, const R* l3a,
	const R* p, R* J_phi, R* J_psi)
{

	R e1[3], e2[3], e3[3];
	minusv3(x1, p, e1);
	minusv3(x2, p, e2);
	minusv3(x3, p, e3);

	R J1[3], J2[3], J3[3];
	crossv3(e3, e2, J1);
	crossv3(e1, e3, J2);
	crossv3(e2, e1, J3);

	R sR1, sR2, sR3;
	sR1 = normv3(e2) + normv3(e3);
	sR2 = normv3(e3) + normv3(e1);
	sR3 = normv3(e1) + normv3(e2);

	R sc1, sc2, sc3;
	sc1 = 1 / l1a[0] * log((sR1 + l1a[0]) / (sR1 - l1a[0]));
	sc2 = 1 / l2a[0] * log((sR2 + l2a[0]) / (sR2 - l2a[0]));
	sc3 = 1 / l3a[0] * log((sR3 + l3a[0]) / (sR3 - l3a[0]));

	R M[3];
	M[0] = dx1[0] * sc1 + dx2[0] * sc2 + dx3[0] * sc3;
	M[1] = dx1[1] * sc1 + dx2[1] * sc2 + dx3[1] * sc3;
	M[2] = dx1[2] * sc1 + dx2[2] * sc2 + dx3[2] * sc3;

	R omega;
	omega = -calc_omega(e1, e2, e3);

	R P[3];
	crossv3(N, M, P);
	R temp[3];
	copyv3(N, temp);
	scalv3(omega, temp);
	addv3(P, temp, P);
	scalv3(1 / (4 * CUDART_PI), P);

	// J_phi & J_psi
	copyv3(P, J_psi);
	scalv3(R(-1), J_psi);

	crossv3(P, dx1, J_phi);
	crossv3(P, dx2, J_phi + 3);
	crossv3(P, dx3, J_phi + 6);

	scalv3(R(1) / nn[0], J_phi);
	scalv3(R(1) / nn[0], J_phi + 3);
	scalv3(R(1) / nn[0], J_phi + 6);
}


template<class R>
__device__ void green_coords_3d_urago3_hessian(const R* x1, const R* x2, const R* x3, const R* N, const R* nn, const R* dx1, const R* dx2, const R* dx3, const R* l1a, const R* l2a, const R* l3a,
	const R* cNdx1, const R* cNdx2, const R* cNdx3, const R* p, R* H_phi, R* H_psi)
{
	R e1[3], e2[3], e3[3];
	minusv3(x1, p, e1);
	minusv3(x2, p, e2);
	minusv3(x3, p, e3);

	R temp[9];
	R* J1 = temp;
	R* J2 = temp + 3;
	R* J3 = temp + 6;
	crossv3(e3, e2, J1);
	crossv3(e1, e3, J2);
	crossv3(e2, e1, J3);

	R sR1, sR2, sR3;
	sR1 = normv3(e2) + normv3(e3);
	sR2 = normv3(e3) + normv3(e1);
	sR3 = normv3(e1) + normv3(e2);

	R oR1, oR2, oR3;
	oR1 = 1 / normv3(e2) + 1 / normv3(e3);
	oR2 = 1 / normv3(e3) + 1 / normv3(e1);
	oR3 = 1 / normv3(e1) + 1 / normv3(e2);

	normalizedv3(e1);
	normalizedv3(e2);
	normalizedv3(e3);

	R dsc1, dsc2, dsc3;
	dsc1 = 2 / ((sR1 + l1a[0]) * (sR1 - l1a[0]));
	dsc2 = 2 / ((sR2 + l2a[0]) * (sR2 - l2a[0]));
	dsc3 = 2 / ((sR3 + l3a[0]) * (sR3 - l3a[0]));

	R Et[9];
	addv3(e2, e3, Et);
	scalv3(dsc1, Et);
	addv3(e3, e1, Et + 3);
	scalv3(dsc2, Et + 3);
	addv3(e1, e2, Et + 6);
	scalv3(dsc3, Et + 6);

	R dscor[3] = { dsc1 * oR1, dsc2 * oR2, dsc3 * oR3 };

	R grad_omega_x, grad_omega_y, grad_omega_z;
	grad_omega_x = J1[0] * dscor[0] + J2[0] * dscor[1] + J3[0] * dscor[2];
	grad_omega_y = J1[1] * dscor[0] + J2[1] * dscor[1] + J3[1] * dscor[2];
	grad_omega_z = J1[2] * dscor[0] + J2[2] * dscor[1] + J3[2] * dscor[2];

	R* J_P = temp;
	J_P[0] = (Et[0] * cNdx1[0] + Et[3] * cNdx2[0] + Et[6] * cNdx3[0] + grad_omega_x * N[0]) / (4 * CUDART_PI);
	J_P[1] = (Et[1] * cNdx1[0] + Et[4] * cNdx2[0] + Et[7] * cNdx3[0] + grad_omega_y * N[0]) / (4 * CUDART_PI);
	J_P[2] = (Et[2] * cNdx1[0] + Et[5] * cNdx2[0] + Et[8] * cNdx3[0] + grad_omega_z * N[0]) / (4 * CUDART_PI);

	J_P[3] = (Et[0] * cNdx1[1] + Et[3] * cNdx2[1] + Et[6] * cNdx3[1] + grad_omega_x * N[1]) / (4 * CUDART_PI);
	J_P[4] = (Et[1] * cNdx1[1] + Et[4] * cNdx2[1] + Et[7] * cNdx3[1] + grad_omega_y * N[1]) / (4 * CUDART_PI);
	J_P[5] = (Et[2] * cNdx1[1] + Et[5] * cNdx2[1] + Et[8] * cNdx3[1] + grad_omega_z * N[1]) / (4 * CUDART_PI);

	J_P[6] = (Et[0] * cNdx1[2] + Et[3] * cNdx2[2] + Et[6] * cNdx3[2] + grad_omega_x * N[2]) / (4 * CUDART_PI);
	J_P[7] = (Et[1] * cNdx1[2] + Et[4] * cNdx2[2] + Et[7] * cNdx3[2] + grad_omega_y * N[2]) / (4 * CUDART_PI);
	J_P[8] = (Et[2] * cNdx1[2] + Et[5] * cNdx2[2] + Et[8] * cNdx3[2] + grad_omega_z * N[2]) / (4 * CUDART_PI);

	// H_phi & H_psi
	H_psi[0] = -J_P[0];
	H_psi[1] = -J_P[1];
	H_psi[2] = -J_P[2];
	H_psi[3] = -J_P[4];
	H_psi[4] = -J_P[5];

	R* temp2 = dscor;
	crossv3(dx1, J_P, temp2);
	H_phi[0] = (-temp2[0]) / nn[0];
	crossv3(dx1, J_P + 3, temp2);
	H_phi[1] = (-temp2[0]) / nn[0];
	H_phi[3] = (-temp2[1]) / nn[0];
	crossv3(dx1, J_P + 6, temp2);
	H_phi[2] = (-temp2[0]) / nn[0];
	H_phi[4] = (-temp2[1]) / nn[0];

	crossv3(dx2, J_P, temp2);
	H_phi[5] = (-temp2[0]) / nn[0];
	crossv3(dx2, J_P + 3, temp2);
	H_phi[6] = (-temp2[0]) / nn[0];
	H_phi[8] = (-temp2[1]) / nn[0];
	crossv3(dx2, J_P + 6, temp2);
	H_phi[7] = (-temp2[0]) / nn[0];
	H_phi[9] = (-temp2[1]) / nn[0];

	crossv3(dx3, J_P, temp2);
	H_phi[10] = (-temp2[0]) / nn[0];
	crossv3(dx3, J_P + 3, temp2);
	H_phi[11] = (-temp2[0]) / nn[0];
	H_phi[13] = (-temp2[1]) / nn[0];
	crossv3(dx3, J_P + 6, temp2);
	H_phi[12] = (-temp2[0]) / nn[0];
	H_phi[14] = (-temp2[1]) / nn[0];
}
