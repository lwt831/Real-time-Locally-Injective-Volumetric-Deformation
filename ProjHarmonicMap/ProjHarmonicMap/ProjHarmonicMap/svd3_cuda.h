/**************************************************************************
**
**  svd3
**
** Quick singular value decomposition as described by:
** A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis,
** Computing the Singular Value Decomposition of 3x3 matrices
** with minimal branching and elementary doubleing point operations,
**  University of Wisconsin - Madison technical report TR1690, May 2011
**
**	OPTIMIZED GPU VERSION
** 	Implementation by: Eric Jang
**
**  13 Apr 2014
**
**************************************************************************/

#ifndef SVD3_CUDA_H
#define SVD3_CUDA_H

#define _gamma 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532 // cos(pi/8)
#define _sstar 0.3826834323 // sin(p/8)
#define EPSILON 1e-10

#include <cuda.h>
#include "cuda_runtime.h"
#include "math.h" // CUDA math library


template<class R>
__host__ __device__ __forceinline__
void condSwap(bool c, R &X, R &Y)
{
	// used in step 2
	R Z = X;
	X = c ? Y : X;
	Y = c ? Z : Y;
}

template<class R>
__host__ __device__ __forceinline__
void condNegSwap(bool c, R &X, R &Y)
{
	// used in step 2 and 3
	R Z = -X;
	X = c ? Y : X;
	Y = c ? Z : Y;
}

// matrix multiplication M = A * B
template<class R>
__host__ __device__ __forceinline__
void multAB(R a11, R a12, R a13,
	R a21, R a22, R a23,
	R a31, R a32, R a33,
	//
	R b11, R b12, R b13,
	R b21, R b22, R b23,
	R b31, R b32, R b33,
	//
	R &m11, R &m12, R &m13,
	R &m21, R &m22, R &m23,
	R &m31, R &m32, R &m33)
{

	m11 = a11 * b11 + a12 * b21 + a13 * b31; m12 = a11 * b12 + a12 * b22 + a13 * b32; m13 = a11 * b13 + a12 * b23 + a13 * b33;
	m21 = a21 * b11 + a22 * b21 + a23 * b31; m22 = a21 * b12 + a22 * b22 + a23 * b32; m23 = a21 * b13 + a22 * b23 + a23 * b33;
	m31 = a31 * b11 + a32 * b21 + a33 * b31; m32 = a31 * b12 + a32 * b22 + a33 * b32; m33 = a31 * b13 + a32 * b23 + a33 * b33;
}

// matrix multiplication M = Transpose[A] * B
template<class R>
__host__ __device__ __forceinline__
void multAtB(R a11, R a12, R a13,
	R a21, R a22, R a23,
	R a31, R a32, R a33,
	//
	R b11, R b12, R b13,
	R b21, R b22, R b23,
	R b31, R b32, R b33,
	//
	R &m11, R &m12, R &m13,
	R &m21, R &m22, R &m23,
	R &m31, R &m32, R &m33)
{
	m11 = a11 * b11 + a21 * b21 + a31 * b31; m12 = a11 * b12 + a21 * b22 + a31 * b32; m13 = a11 * b13 + a21 * b23 + a31 * b33;
	m21 = a12 * b11 + a22 * b21 + a32 * b31; m22 = a12 * b12 + a22 * b22 + a32 * b32; m23 = a12 * b13 + a22 * b23 + a32 * b33;
	m31 = a13 * b11 + a23 * b21 + a33 * b31; m32 = a13 * b12 + a23 * b22 + a33 * b32; m33 = a13 * b13 + a23 * b23 + a33 * b33;
}

template<class R>
__host__ __device__ __forceinline__
void quatToMat3(const R * qV,
	R &m11, R &m12, R &m13,
	R &m21, R &m22, R &m23,
	R &m31, R &m32, R &m33
)
{
	R w = qV[3];
	R x = qV[0];
	R y = qV[1];
	R z = qV[2];

	R qxx = x * x;
	R qyy = y * y;
	R qzz = z * z;
	R qxz = x * z;
	R qxy = x * y;
	R qyz = y * z;
	R qwx = w * x;
	R qwy = w * y;
	R qwz = w * z;

	m11 = 1 - 2 * (qyy + qzz); m12 = 2 * (qxy - qwz); m13 = 2 * (qxz + qwy);
	m21 = 2 * (qxy + qwz); m22 = 1 - 2 * (qxx + qzz); m23 = 2 * (qyz - qwx);
	m31 = 2 * (qxz - qwy); m32 = 2 * (qyz + qwx); m33 = 1 - 2 * (qxx + qyy);
}

template<class R>
__host__ __device__ __forceinline__
void approximateGivensQuaternion(R a11, R a12, R a22, R &ch, R &sh)
{
	/*
		 * Given givens angle computed by approximateGivensAngles,
		 * compute the corresponding rotation quaternion.
		 */
	ch = 2 * (a11 - a22);
	sh = a12;
	bool b = _gamma * sh*sh < ch*ch;
	R w = rsqrt(ch*ch + sh * sh);
	ch = b ? w * ch : _cstar;
	sh = b ? w * sh : _sstar;
}

template<class R>
__host__ __device__ __forceinline__
void jacobiConjugation(const int x, const int y, const int z,
	R &s11,
	R &s21, R &s22,
	R &s31, R &s32, R &s33,
	R * qV)
{
	R ch, sh;
	approximateGivensQuaternion(s11, s21, s22, ch, sh);

	R scale = ch * ch + sh * sh;
	R a = (ch*ch - sh * sh) / scale;
	R b = (2 * sh*ch) / scale;

	// make temp copy of S
	R _s11 = s11;
	R _s21 = s21; R _s22 = s22;
	R _s31 = s31; R _s32 = s32; R _s33 = s33;

	// perform conjugation S = Q'*S*Q
	// Q already implicitly solved from a, b
	s11 = a * (a*_s11 + b * _s21) + b * (a*_s21 + b * _s22);
	s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);	s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
	s31 = a * _s31 + b * _s32;	s32 = -b * _s31 + a * _s32; s33 = _s33;

	// update cumulative rotation qV
	R tmp[3];
	tmp[0] = qV[0] * sh;
	tmp[1] = qV[1] * sh;
	tmp[2] = qV[2] * sh;
	sh *= qV[3];

	qV[0] *= ch;
	qV[1] *= ch;
	qV[2] *= ch;
	qV[3] *= ch;

	// (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
	// for (p,q) = ((0,1),(1,2),(0,2))
	qV[z] += sh;
	qV[3] -= tmp[z]; // w
	qV[x] += tmp[y];
	qV[y] -= tmp[x];

	// re-arrange matrix for next iteration
	_s11 = s22;
	_s21 = s32; _s22 = s33;
	_s31 = s21; _s32 = s31; _s33 = s11;
	s11 = _s11;
	s21 = _s21; s22 = _s22;
	s31 = _s31; s32 = _s32; s33 = _s33;

}

template<class R>
__host__ __device__ __forceinline__
R dist2(R x, R y, R z)
{
	return x * x + y * y + z * z;
}

// finds transformation that diagonalizes a symmetric matrix
template<class R>
__host__ __device__ __forceinline__
void jacobiEigenanlysis( // symmetric matrix
	R &s11,
	R &s21, R &s22,
	R &s31, R &s32, R &s33,
	// quaternion representation of V
	R * qV)
{
	qV[3] = 1; qV[0] = 0; qV[1] = 0; qV[2] = 0; // follow same indexing convention as GLM
	for (int i = 0; i < 10; i++)
	{
		// we wish to eliminate the maximum off-diagonal element
		// on every iteration, but cycling over all 10 possible rotations
		// in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
		//  asymptotic convergence
		jacobiConjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,1
		jacobiConjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV); // p,q = 1,2
		jacobiConjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,2
	}
}

template<class R>
__host__ __device__ __forceinline__
void sortSingularValues(// matrix that we want to decompose
	R &b11, R &b12, R &b13,
	R &b21, R &b22, R &b23,
	R &b31, R &b32, R &b33,
	// sort V simultaneously
	R &v11, R &v12, R &v13,
	R &v21, R &v22, R &v23,
	R &v31, R &v32, R &v33)
{
	R rho1 = dist2(b11, b21, b31);
	R rho2 = dist2(b12, b22, b32);
	R rho3 = dist2(b13, b23, b33);
	bool c;
	c = rho1 < rho2;
	condNegSwap(c, b11, b12); condNegSwap(c, v11, v12);
	condNegSwap(c, b21, b22); condNegSwap(c, v21, v22);
	condNegSwap(c, b31, b32); condNegSwap(c, v31, v32);
	condSwap(c, rho1, rho2);
	c = rho1 < rho3;
	condNegSwap(c, b11, b13); condNegSwap(c, v11, v13);
	condNegSwap(c, b21, b23); condNegSwap(c, v21, v23);
	condNegSwap(c, b31, b33); condNegSwap(c, v31, v33);
	condSwap(c, rho1, rho3);
	c = rho2 < rho3;
	condNegSwap(c, b12, b13); condNegSwap(c, v12, v13);
	condNegSwap(c, b22, b23); condNegSwap(c, v22, v23);
	condNegSwap(c, b32, b33); condNegSwap(c, v32, v33);
}

template<class R>
__host__ __device__ __forceinline__
void QRGivensQuaternion(R a1, R a2, R &ch, R &sh)
{
	// a1 = pivot point on diagonal
	// a2 = lower triangular entry we want to annihilate
	R epsilon = EPSILON;
	R rho = sqrt(a1*a1 + a2 * a2);

	sh = rho > epsilon ? a2 : 0;
	ch = fabs(a1) + fmax(rho, epsilon);
	bool b = a1 < 0;
	condSwap(b, sh, ch);
	R w = rsqrt(ch*ch + sh * sh);
	ch *= w;
	sh *= w;
}

template<class R>
__host__ __device__ __forceinline__
void QRDecomposition(// matrix that we want to decompose
	R b11, R b12, R b13,
	R b21, R b22, R b23,
	R b31, R b32, R b33,
	// output Q
	R &q11, R &q12, R &q13,
	R &q21, R &q22, R &q23,
	R &q31, R &q32, R &q33,
	// output R
	R &r11, R &r12, R &r13,
	R &r21, R &r22, R &r23,
	R &r31, R &r32, R &r33)
{
	R ch1, sh1, ch2, sh2, ch3, sh3;
	R a, b;

	// first givens rotation (ch,0,0,sh)
	QRGivensQuaternion(b11, b21, ch1, sh1);
	a = 1 - 2 * sh1*sh1;
	b = 2 * ch1*sh1;
	// apply B = Q' * B
	r11 = a * b11 + b * b21;  r12 = a * b12 + b * b22;  r13 = a * b13 + b * b23;
	r21 = -b * b11 + a * b21; r22 = -b * b12 + a * b22; r23 = -b * b13 + a * b23;
	r31 = b31;          r32 = b32;          r33 = b33;

	// second givens rotation (ch,0,-sh,0)
	QRGivensQuaternion(r11, r31, ch2, sh2);
	a = 1 - 2 * sh2*sh2;
	b = 2 * ch2*sh2;
	// apply B = Q' * B;
	b11 = a * r11 + b * r31;  b12 = a * r12 + b * r32;  b13 = a * r13 + b * r33;
	b21 = r21;           b22 = r22;           b23 = r23;
	b31 = -b * r11 + a * r31; b32 = -b * r12 + a * r32; b33 = -b * r13 + a * r33;

	// third givens rotation (ch,sh,0,0)
	QRGivensQuaternion(b22, b32, ch3, sh3);
	a = 1 - 2 * sh3*sh3;
	b = 2 * ch3*sh3;
	// R is now set to desired value
	r11 = b11;             r12 = b12;           r13 = b13;
	r21 = a * b21 + b * b31;     r22 = a * b22 + b * b32;   r23 = a * b23 + b * b33;
	r31 = -b * b21 + a * b31;    r32 = -b * b22 + a * b32;  r33 = -b * b23 + a * b33;

	// construct the cumulative rotation Q=Q1 * Q2 * Q3
	// the number of Ring point operations for three quaternion multiplications
	// is more or less comparable to the explicit form of the joined matrix.
	// certainly more memory-efficient!
	R sh12 = sh1 * sh1;
	R sh22 = sh2 * sh2;
	R sh32 = sh3 * sh3;

	q11 = (-1 + 2 * sh12)*(-1 + 2 * sh22);
	q12 = 4 * ch2*ch3*(-1 + 2 * sh12)*sh2*sh3 + 2 * ch1*sh1*(-1 + 2 * sh32);
	q13 = 4 * ch1*ch3*sh1*sh3 - 2 * ch2*(-1 + 2 * sh12)*sh2*(-1 + 2 * sh32);

	q21 = 2 * ch1*sh1*(1 - 2 * sh22);
	q22 = -8 * ch1*ch2*ch3*sh1*sh2*sh3 + (-1 + 2 * sh12)*(-1 + 2 * sh32);
	q23 = -2 * ch3*sh3 + 4 * sh1*(ch3*sh1*sh3 + ch1 * ch2*sh2*(-1 + 2 * sh32));

	q31 = 2 * ch2*sh2;
	q32 = 2 * ch3*(1 - 2 * sh22)*sh3;
	q33 = (-1 + 2 * sh22)*(-1 + 2 * sh32);
}

template<class R>
__host__ __device__ __forceinline__
void svd(// input A
	R a11, R a12, R a13,
	R a21, R a22, R a23,
	R a31, R a32, R a33,
	// output U
	R &u11, R &u12, R &u13,
	R &u21, R &u22, R &u23,
	R &u31, R &u32, R &u33,
	// output S
	R &s11, R &s12, R &s13,
	R &s21, R &s22, R &s23,
	R &s31, R &s32, R &s33,
	// output V
	R &v11, R &v12, R &v13,
	R &v21, R &v22, R &v23,
	R &v31, R &v32, R &v33)
{
	// normal equations matrix
	R ATA11, ATA12, ATA13;
	R ATA21, ATA22, ATA23;
	R ATA31, ATA32, ATA33;

	multAtB(a11, a12, a13, a21, a22, a23, a31, a32, a33,
		a11, a12, a13, a21, a22, a23, a31, a32, a33,
		ATA11, ATA12, ATA13, ATA21, ATA22, ATA23, ATA31, ATA32, ATA33);

	// symmetric eigenalysis
	R qV[4];
	jacobiEigenanlysis(ATA11, ATA21, ATA22, ATA31, ATA32, ATA33, qV);
	quatToMat3(qV, v11, v12, v13, v21, v22, v23, v31, v32, v33);

	R b11, b12, b13;
	R b21, b22, b23;
	R b31, b32, b33;
	multAB(a11, a12, a13, a21, a22, a23, a31, a32, a33,
		v11, v12, v13, v21, v22, v23, v31, v32, v33,
		b11, b12, b13, b21, b22, b23, b31, b32, b33);

	// sort singular values and find V
	sortSingularValues(b11, b12, b13, b21, b22, b23, b31, b32, b33,
		v11, v12, v13, v21, v22, v23, v31, v32, v33);

	// QR decomposition
	QRDecomposition(b11, b12, b13, b21, b22, b23, b31, b32, b33,
		u11, u12, u13, u21, u22, u23, u31, u32, u33,
		s11, s12, s13, s21, s22, s23, s31, s32, s33
	);
}

/// polar decomposition can be reconstructed trivially from SVD result
/// A = UP
template<class R>
__host__ __device__ __forceinline__
void pd(R a11, R a12, R a13,
	R a21, R a22, R a23,
	R a31, R a32, R a33,
	// output U
	R &u11, R &u12, R &u13,
	R &u21, R &u22, R &u23,
	R &u31, R &u32, R &u33,
	// output P
	R &p11, R &p12, R &p13,
	R &p21, R &p22, R &p23,
	R &p31, R &p32, R &p33)
{
	R w11, w12, w13, w21, w22, w23, w31, w32, w33;
	R s11, s12, s13, s21, s22, s23, s31, s32, s33;
	R v11, v12, v13, v21, v22, v23, v31, v32, v33;

	svd(a11, a12, a13, a21, a22, a23, a31, a32, a33,
		w11, w12, w13, w21, w22, w23, w31, w32, w33,
		s11, s12, s13, s21, s22, s23, s31, s32, s33,
		v11, v12, v13, v21, v22, v23, v31, v32, v33);

	// P = VSV'
	R t11, t12, t13, t21, t22, t23, t31, t32, t33;
	multAB(v11, v12, v13, v21, v22, v23, v31, v32, v33,
		s11, s12, s13, s21, s22, s23, s31, s32, s33,
		t11, t12, t13, t21, t22, t23, t31, t32, t33);

	multAB(t11, t12, t13, t21, t22, t23, t31, t32, t33,
		v11, v21, v31, v12, v22, v32, v13, v23, v33,
		p11, p12, p13, p21, p22, p23, p31, p32, p33);

	// U = WV'
	multAB(w11, w12, w13, w21, w22, w23, w31, w32, w33,
		v11, v21, v31, v12, v22, v32, v13, v23, v33,
		u11, u12, u13, u21, u22, u23, u31, u32, u33);
}

#endif
