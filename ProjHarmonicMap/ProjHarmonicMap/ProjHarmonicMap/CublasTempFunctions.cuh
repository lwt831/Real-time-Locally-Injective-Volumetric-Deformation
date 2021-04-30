#pragma once
#include<cublas_v2.h>
#include<cusolverDn.h>

/*-----cublas<t>fun-----*/
//scal: x = alpha*x
inline cublasStatus_t scal(cublasHandle_t handle,
	int n,
	const double* alpha,
	double* x,
	int incx)
{
	return cublasDscal(handle, n, alpha, x, incx);
}
inline cublasStatus_t scal(cublasHandle_t handle, 
	int n,
	const float* alpha,
	float* x, 
	int incx)
{
	return cublasSscal(handle, n, alpha, x, incx);
}


//nrm2: result = norm(x)^2
inline cublasStatus_t nrm2(cublasHandle_t handle,
	int n,
	const double* x,
	int incx,
	double* result
)
{
	return cublasDnrm2(handle, n, x, incx, result);
}
inline cublasStatus_t nrm2(cublasHandle_t handle,
	int n,
	const float* x,
	int incx,
	float* result
)
{
	return cublasSnrm2(handle, n, x, incx, result);
}

//dot: result = x dot y
inline cublasStatus_t dot(cublasHandle_t handle,
	int n,
	const double* x,
	int incx,
	const double* y,
	int incy,
	double* result
)
{
	return cublasDdot(handle, n, x, incx, y, incy, result);
}
inline cublasStatus_t dot(cublasHandle_t handle,
	int n,
	const float* x,
	int incx,
	const float* y,
	int incy,
	float* result
)
{
	return cublasSdot(handle, n, x, incx, y, incy, result);
}

//gemv: y=alpha*A*x+beta*y
inline cublasStatus_t gemv(cublasHandle_t handle,
	cublasOperation_t trans,
	int m,
	int n,
	const double *alpha, /* host or device pointer */
	const double *A,
	int lda,
	const double *x,
	int incx,
	const double *beta, /* host or device pointer */
	double *y,
	int incy)
{

	return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
inline cublasStatus_t gemv(cublasHandle_t handle,
	cublasOperation_t trans,
	int m,
	int n,
	const float *alpha, /* host or device pointer */
	const float *A,
	int lda,
	const float *x,
	int incx,
	const float *beta, /* host or device pointer */
	float *y,
	int incy)
{

	return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

//asum: result=sum(x)
inline cublasStatus_t asum(cublasHandle_t handle,
	int n,
	const double *x,
	int incx,
	double *result)
{
	return cublasDasum(handle, n, x, incx, result);
}
inline cublasStatus_t asum(cublasHandle_t handle,
	int n,
	const float *x,
	int incx,
	float *result)
{
	return cublasSasum(handle, n, x, incx, result);
}

//gemm: C = alpha * A*B + beta * C
inline cublasStatus_t gemm(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	const double *alpha, /* host or device pointer */
	const double *A,
	int lda,
	const double *B,
	int ldb,
	const double *beta, /* host or device pointer */
	double *C,
	int ldc)
{
	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t gemm(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	const float *alpha, /* host or device pointer */
	const float *A,
	int lda,
	const float *B,
	int ldb,
	const float *beta, /* host or device pointer */
	float *C,
	int ldc)
{
	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

//syrk: C = alpha * A*A' + beta * C (symmetric matrix update rank k)
inline cublasStatus_t syrk(cublasHandle_t handle,
	cublasFillMode_t uplo,
	cublasOperation_t trans,
	int n,
	int k,
	const double* alpha, /* host or device pointer */
	const double* A,
	int lda,
	const double* beta, /* host or device pointer */
	double* C,
	int ldc)
{
	return cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
inline cublasStatus_t syrk(cublasHandle_t handle,
	cublasFillMode_t uplo,
	cublasOperation_t trans,
	int n,
	int k,
	const float* alpha, /* host or device pointer */
	const float* A,
	int lda,
	const float* beta, /* host or device pointer */
	float* C,
	int ldc)
{
	return cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}


//axpy: C=A+B	
inline cublasStatus_t axpy(cublasHandle_t handle,
	int n,
	const double *alpha, /* host or device pointer */
	const double *x,
	int incx,
	double *y,
	int incy)
{
	return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t axpy(cublasHandle_t handle,
	int n,
	const float *alpha, /* host or device pointer */
	const float *x,
	int incx,
	float *y,
	int incy)
{
	return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

//dgmm: C=diag(X)*B or C=B*diag(X)
inline cublasStatus_t dgmm(cublasHandle_t handle,
	cublasSideMode_t mode,
	int m,
	int n,
	const double *A,
	int lda,
	const double *x,
	int incx,
	double *C,
	int ldc)
{
	return cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}
inline cublasStatus_t dgmm(cublasHandle_t handle,
	cublasSideMode_t mode,
	int m,
	int n,
	const float *A,
	int lda,
	const float *x,
	int incx,
	float *C,
	int ldc)
{
	return cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

//geam: C=alpha*op(A)+beta*op(B)  //using the function to get transpose of matrix A by setting alpha=1,beta=0,op(A)=tran
inline cublasStatus_t geam(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	const double *alpha, /* host or device pointer */
	const double *A,
	int lda,
	const double *beta, /* host or device pointer */
	const double *B,
	int ldb,
	double *C,
	int ldc)
{
	return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t geam(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	const float *alpha, /* host or device pointer */
	const float *A,
	int lda,
	const float *beta, /* host or device pointer */
	const float *B,
	int ldb,
	float *C,
	int ldc)
{
	return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

//gemmBatched: C[i]=A[i]*B[i]
inline cublasStatus_t gemmBatched(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	const double *alpha,  /* host or device pointer */
	const double *const Aarray[],
	int lda,
	const double *const Barray[],
	int ldb,
	const double *beta,  /* host or device pointer */
	double *const Carray[],
	int ldc,
	int batchCount)
{
	return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline cublasStatus_t gemmBatched(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	const float *alpha,  /* host or device pointer */
	const float *const Aarray[],
	int lda,
	const float *const Barray[],
	int ldb,
	const float *beta,  /* host or device pointer */
	float *const Carray[],
	int ldc,
	int batchCount)
{
	return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

/*-----cusolverDn<t>fun-----*/

//LUSolver
inline cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t handle,
int m,
int n,
double *A,
int lda,
int *Lwork )
{
	return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}
inline cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t handle,
	int m,
	int n,
	float *A,
	int lda,
	int *Lwork)
{
	return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

inline cusolverStatus_t getrf(cusolverDnHandle_t handle,
	int m,
	int n,
	double *A,
	int lda,
	double *Workspace,
	int *devIpiv,
	int *devInfo)
{
	return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}
inline cusolverStatus_t getrf(cusolverDnHandle_t handle,
	int m,
	int n,
	float *A,
	int lda,
	float *Workspace,
	int *devIpiv,
	int *devInfo)
{
	return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

inline cusolverStatus_t getrs(cusolverDnHandle_t handle,
	cublasOperation_t trans,
	int n,
	int nrhs,
	const double *A,
	int lda,
	const int *devIpiv,
	double *B,
	int ldb,
	int *devInfo)
{
	return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}
inline cusolverStatus_t getrs(cusolverDnHandle_t handle,
	cublasOperation_t trans,
	int n,
	int nrhs,
	const float *A,
	int lda,
	const int *devIpiv,
	float *B,
	int ldb,
	int *devInfo)
{
	return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

//CholeskySolver
inline cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t handle,
	cublasFillMode_t uplo,
	int n,
	double *A,
	int lda,
	int *Lwork)
{
	return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}
inline cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t handle,
	cublasFillMode_t uplo,
	int n,
	float *A,
	int lda,
	int *Lwork)
{
	return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

inline cusolverStatus_t potrf(cusolverDnHandle_t handle,
	cublasFillMode_t uplo,
	int n,
	double *A,
	int lda,
	double *Workspace,
	int Lwork,
	int *devInfo)
{
	return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}
inline cusolverStatus_t potrf(cusolverDnHandle_t handle,
	cublasFillMode_t uplo,
	int n,
	float *A,
	int lda,
	float *Workspace,
	int Lwork,
	int *devInfo)
{
	return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

inline cusolverStatus_t potrs(cusolverDnHandle_t handle,
	cublasFillMode_t uplo,
	int n,
	int nrhs,
	const double *A,
	int lda,
	double *B,
	int ldb,
	int *devInfo)
{
	return cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}
inline cusolverStatus_t potrs(cusolverDnHandle_t handle,
	cublasFillMode_t uplo,
	int n,
	int nrhs,
	const float *A,
	int lda,
	float *B,
	int ldb,
	int *devInfo)
{
	return cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

////using real = double;
////using real = float;
//
//#define real_Double
////define real_Single
//
//#ifdef real_Double
//
///*cublas<t>fun*/
//#define gemv cublasDgemv	//y=alpha*A*x+beta*y
//#define asum cublasDasum	//result=sum(A)
//#define gemm cublasDgemm	//C=A*B
//#define axpy cublasDaxpy	//C=A+B	
//#define dgmm cublasDdgmm	//C=diag(X)*B or C=B*diag(X)
//#define geam cublasDgeam	//C=alpha*op(A)+beta*op(B)  //using the function to get transpose of matrix A by setting alpha=1,beta=0,op(A)=tran
//#define gemmBatched cublasDgemmBatched	//C[i]=A[i]*B[i]
//#define dot cublasDdot
//#define nrm2 cublasDnrm2
//#define scal cublasDscal
//
//cublasStatus_t minv(
//		cublasHandle_t handle,
//		int n,
//		const double *x,
//		int incx,
//		double& result)
//	{
//		int min_index;
//		cublasStatus_t cuStatus_t;
//		cuStatus_t = cublasIdamin(handle, n, x, incx, &min_index);
//		cudaMemcpy(&result, x + min_index - 1, sizeof(double), cudaMemcpyDeviceToHost);
//		return cuStatus_t;
//	}
//
//
///*cusolverDn<t>fun*/
//
////LuSolver
//#define getrf_bufferSize cusolverDnDgetrf_bufferSize
//#define getrf cusolverDnDgetrf
//#define getrs cusolverDnDgetrs
//
////CholeskySolver
//#define potrf_bufferSize cusolverDnDpotrf_bufferSize
//#define potrf cusolverDnDpotrf
//#define potrs cusolverDnDpotrs
//#endif
//
//#ifdef real_Single
///*cublas<t>fun*/
//#define gemv cublasSgemv
//#define asum cublasSasum
//#define gemm cublasSgemm
//#define axpy cublasSaxpy
//#define dgmm cublasSdgmm
//#define geam cublasSgeam
//#define gemmBatched cublasSgemmBatched
//
///*cusolverDn<t>fun*/
////LuSolver
//#define getrf_bufferSize cusolverDnSgetrf_bufferSize
//#define getrf cusolverDnSgetrf
//#define getrs cusolverDnSgetrs
//
////CholeskySolver
//#define potrf_bufferSize cusolverDnSpotrf_bufferSize
//#define potrf cusolverDnSpotrf
//#define potrs cusolverDnSpotrs
//
//#endif
//
//#undef real_Double
////undef real_Single





