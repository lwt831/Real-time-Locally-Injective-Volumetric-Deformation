#pragma once
#include"CublasTempFunctions.cuh"
#include<cuda_runtime.h>
#include<cusolverDn.h>
#include<vector>
#include<cublas_v2.h>
#include <algorithm>




const int threadsPerBlock = 256;
inline __host__ __device__ int ceildiv(int m, int n) { return (m - 1) / n + 1; }
inline __host__ __device__ int blockNum(int n) { return ceildiv(n, threadsPerBlock); }

template<class R, int n = 1>
__host__ __device__ void print_gpu_value(const R* d_v, const char* valname = "value", bool newline = true)
{
#if defined(_DEBUG) && !defined(__CUDA_ARCH__)      // do not call the following on cuda kernel
	R v[n];
	cudaMemcpy(v, d_v, sizeof(R) * n, cudaMemcpyDeviceToHost);
	printf("%20s = ", valname);
	for (int i = 0; i < n; i++)
		printf("%10.9e%s", v[i], (i < n - 1) ? "\t" : (newline ? "\n" : ""));
#endif
}



#if defined(_DEBUG) && !defined(__CUDACC__)
#define CUDA_CHECK_ERROR cudaCheckError(__FILE__, __LINE__);
#else
#define CUDA_CHECK_ERROR  
#endif

template<class T>
inline cudaError_t myCopy_n(const T *src, int n, T *dst, cudaMemcpyKind cpydir = cudaMemcpyDeviceToDevice) { return cudaMemcpyAsync(dst, src, sizeof(T)*n, cpydir); }

template< typename T >
void checkCuda(T result, char const *const func, const char *const file, int const line)
{
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
		cudaDeviceReset();

		// Make sure we call CUDA Device Reset before exiting
		throw std::exception("cuda runtime error");
	}
}

#define checkCudaErrors(val)           checkCuda ( (val), #val, __FILE__, __LINE__ )
#define ensure(cond, msg) if (!(cond)) fprintf(stderr, (msg+std::string(" at file %s, line %d\n")).c_str(), __FILE__, __LINE__);

__forceinline__ __device__ void atomicMinPositive(float *t, float x)
{
	//atomicMin((unsigned int*)t,  __float_as_uint(max(0., x))); // todo: atomicMin for float?  the current code works only if all numbers are non-negative
	atomicMin((unsigned int*)t, __float_as_uint(x));
}

__forceinline__ __device__ void atomicMaxPositive(float *t, float x)
{
	atomicMax((unsigned int*)t, __float_as_uint(x));
}

//https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
// For all float & double atomics:
//      Must do the compare with integers, not floating point,
//      since NaN is never equal to any other NaN
//
// float atomicMin
__device__ __forceinline__ float atomicMin(float *address, float val)
{
	int ret = __float_as_int(*address);
	while (val < __int_as_float(ret))
	{
		int old = ret;
		if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
			break;
	}
	return __int_as_float(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
	int ret = __float_as_int(*address);
	while (val > __int_as_float(ret))
	{
		int old = ret;
		if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
			break;
	}
	return __int_as_float(ret);
}



#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


#if __CUDA_ARCH__ >= 130

__forceinline__ __device__ void atomicMinPositive(double *t, double x)
{
	//atomicMin((unsigned long long*)t, __double_as_longlong(max(0., x)));
	atomicMin((unsigned long long*)t, __double_as_longlong(x));
}

__forceinline__ __device__ void atomicMaxPositive(double *t, double x)
{
	atomicMax((unsigned long long*)t, __double_as_longlong(x));
}

// double atomicMin
__device__ __forceinline__ double atomicMin(double *address, double val)
{
	unsigned long long ret = __double_as_longlong(*address);
	while (val < __longlong_as_double(ret))
	{
		unsigned long long old = ret;
		if ((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
			break;
	}
	return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val)
{
	unsigned long long ret = __double_as_longlong(*address);
	while (val > __longlong_as_double(ret))
	{
		unsigned long long old = ret;
		if ((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
			break;
	}
	return __longlong_as_double(ret);
}
#endif

template<class R>
__global__  void cuVector_set_value(R* pdata, R* scal, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n)
	{
		pdata[tid] = scal[0];
	}
}



enum DataType
{
	Unknow = 0,
	Host = 1,
	Device = 2
};



template<class R>
struct cuVector
{
	R* pdata = nullptr;
	int len;

	cuVector(const cuVector& B)
	{
		clear();
		if (B.len > 0)
		{
			len = B.len;
			checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(R)*len));
			checkCudaErrors(cudaMemcpyAsync(pdata, B.pdata, sizeof(R)*len, cudaMemcpyDeviceToDevice));
		}
	}
	cuVector& operator=(const cuVector& B)
	{

		clear();
		len = B.len;
		if (len != 0)
		{
			checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(R)*len));
			checkCudaErrors(cudaMemcpyAsync(pdata, B.pdata, sizeof(R)*len, cudaMemcpyDeviceToDevice));
		}


		return *this;
	}
	cuVector(cuVector&& t) :len(0), pdata(nullptr) { std::swap(pdata, t.pdata); std::swap(len, t.len); }

	cuVector(int n = 0, const R* srcbegin = nullptr, DataType type = Unknow) :len(n), pdata(nullptr) {
		if (len > 0) {
			checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(R)*len));
			if (srcbegin)
			{
				if (type == Host)
					checkCudaErrors(cudaMemcpyAsync(pdata, srcbegin, len * sizeof(R), cudaMemcpyHostToDevice));
				if (type == Device)
					checkCudaErrors(cudaMemcpyAsync(pdata, srcbegin, len * sizeof(R), cudaMemcpyDeviceToDevice));
			}
		}
	}

	cuVector(const std::vector<R> &v) :len(v.size()), pdata(nullptr) {
		if (len > 0) {
			checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(R)*len));
			checkCudaErrors(cudaMemcpyAsync(pdata, v.data(), len * sizeof(R), cudaMemcpyHostToDevice));
		}
	}

	~cuVector() { if (pdata) checkCudaErrors(cudaFree(pdata)); }

	void set_pdata(R*& data, size_t n)
	{
		clear();
		pdata = data;
		len = n;
		data = nullptr;
	}

	void set_value(R *scal) //scal is a point on device
	{
		if(len > 0)
			cuVector_set_value << <blockNum(len), threadsPerBlock >> > (pdata, scal, len);
	}

	int size() const { return len; }
	bool empty() const { return len == 0; }

	void clear() {
		if (pdata) {
			checkCudaErrors(cudaFree(pdata));
			pdata = nullptr;
			len = 0;
		}
	}

	R* data() { return pdata; }
	const R* data() const { return pdata; }

	cuVector& operator =(const std::vector<R> &v) {
		resize(v.size(), false);
		checkCudaErrors(cudaMemcpyAsync(pdata, v.data(), len * sizeof(R), cudaMemcpyHostToDevice));
		return *this;
	}

	cuVector& operator =(cuVector&& t) { std::swap(pdata, t.pdata); std::swap(len, t.len); return *this; }


	void zero_fill() { checkCudaErrors(cudaMemsetAsync(pdata, 0, sizeof(R)*len)); }

	void to_host(R* dst, int n) const {
		assert(n <= len);
		checkCudaErrors(cudaMemcpyAsync(dst, pdata, n * sizeof(R), cudaMemcpyDeviceToHost));
	}

	operator std::vector<R>() const { std::vector<R> v(len); to_host(v.data(), len); return v; }

	void resize(int n, bool copyData = false) {
		if (n == len) return;
		cuVector<R> v2(n);
		if (copyData && len > 0) checkCudaErrors(cudaMemcpyAsync(v2.data(), pdata, std::min(len, n) * sizeof(R), cudaMemcpyDeviceToDevice));
		std::swap(*this, v2);
	}

};


template<class R >
struct cuSolverDN
{
	cusolverDnHandle_t handle;
	cuVector<R> buffer;
	cuVector<R> A;
	cuVector<int> info;
	int n = 0;

	cuSolverDN() :handle(nullptr), info(1) { info.zero_fill(); }
	virtual ~cuSolverDN() { if (handle)   cusolverDnDestroy(handle); }

	void init(int neq) { n = neq; A.resize(n*n, false); }

	virtual int factor(const R* A0 = nullptr) = 0;
	virtual int solve(R *const) = 0;
};




template<class R >
struct cuLUSolverDn : public cuSolverDN<R>
{
	cuVector<int> ipiv; // pivot for LU solver

	cuLUSolverDn() {}
	~cuLUSolverDn() {}

	int factor(const R* A0 = nullptr) {
		if (!handle)   cusolverDnCreate(&handle);


		A.resize(n * n, false);   // allocate memory if necessary
		if (A0)  myCopy_n(A0, n * n, A.data());

		int bufferSize = 0;
		getrf_bufferSize(handle, n, n, (R*)A.data(), n, &bufferSize);

		buffer.resize(bufferSize, false);
		ipiv.resize(n, false);

		//cusolverDnZgetrf
		getrf(handle, n, n, (R*)A.data(), n, (R*)buffer.data(), ipiv.data(), info.data());
		CUDA_CHECK_ERROR;

		return 0;
	}

	int solve(R *const b) {


		// b will be rewritten in place on return
		getrs(handle, CUBLAS_OP_N, n, 1, (R*)A.data(), n, ipiv.data(), (R*)b, n, info.data());

		return 0;
	}
};


template<class R >
struct cuCholSolverDn : public cuSolverDN<R>
{
	static const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	cuCholSolverDn() {}
	virtual ~cuCholSolverDn() {}

	int factor(const R* A0 = nullptr) {
		if (!handle)   cusolverDnCreate(&handle);



		A.resize(n * n, false);   // allocate memory if necessary
		if (A0 && A0 != A.data())  myCopy_n(A0, n * n, A.data());

		int bufferSize = 0;
		potrf_bufferSize(handle, uplo, n, (R*)A.data(), n, &bufferSize);

		buffer.resize(bufferSize, false);

		potrf(handle, uplo, n, (R*)A.data(), n, (R*)buffer.data(), bufferSize, info.data());
		CUDA_CHECK_ERROR;

		return 0;
	}

	int solve(R *const b) {


		// b will be rewritten in place on return
		potrs(handle, uplo, n, 1, (R*)A.data(), n, (R*)b, n, info.data());
		return 0;
	}
};



template<class R>
struct cuMatrix
{

	R* pdata = nullptr;
	size_t dims[2];


	cuSolverDN<R>* solver = nullptr;





	cuMatrix(const cuMatrix& B)
	{

		clear();
		dims[0] = B.dims[0];
		dims[1] = B.dims[1];
		int len = dims[0] * dims[1];
		if (len != 0)
		{
			checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(R)*len));
			checkCudaErrors(cudaMemcpyAsync(pdata, B.pdata, sizeof(R)*len, cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMalloc((void**)&pdata, sizeof(R) * len));
			checkCudaErrors(cudaMemcpyAsync(pdata, B.pdata, sizeof(R) * len, cudaMemcpyDeviceToDevice));
		}


	}


	cuMatrix& operator=(const cuMatrix& B)
	{

		clear();
		dims[0] = B.dims[0];
		dims[1] = B.dims[1];
		int len = dims[0] * dims[1];
		if (len != 0)
		{
			checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(R)*len));
			checkCudaErrors(cudaMemcpyAsync(pdata, B.pdata, sizeof(R)*len, cudaMemcpyDeviceToDevice));
		}


		return *this;
	}
	cuMatrix(cuVector<R> A, int n = 1)
	{
		int m = A.len / n;
		ensure(m*n == A.len, "check size!");
		dims[0] = m; dims[1] = n;
		checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(R)*A.len));
		checkCudaErrors(cudaMemcpyAsync(pdata, A.data(), A.len * sizeof(R), cudaMemcpyDeviceToDevice));
	}

	cuMatrix(int m = 0, int n = 0, DataType type = Unknow, R* data = nullptr)
	{
		dims[0] = m; dims[1] = n;
		int len = m * n;
		if (len > 0)
		{
			checkCudaErrors(cudaMalloc((void**)&pdata, len * sizeof(R)));
		}

		if (data)
		{
			if (type == Host)
			{
				checkCudaErrors(cudaMemcpyAsync(pdata, data, m * n * sizeof(R), cudaMemcpyHostToDevice));
			}
			if (type == Device)
			{
				checkCudaErrors(cudaMemcpyAsync(pdata, data, m * n * sizeof(R), cudaMemcpyDeviceToDevice));
			}
		}
	}

	~cuMatrix() { if (pdata) checkCudaErrors(cudaFree(pdata)); if (solver) delete solver; }

	void zero_fill() { checkCudaErrors(cudaMemsetAsync(pdata, 0, sizeof(R) * dims[0] * dims[1])); }


	const size_t* size() { return (const size_t*)dims; }

	//bool empty() const { return len == 0; }

	bool IsSameSize(const cuMatrix<R> B)
	{
		if (dims[0] == B.dims[0] && dims[1] == B.dims[1])return true;
		else return false;
	}

	void clear() {
		if (pdata) {
			checkCudaErrors(cudaFree(pdata));
			pdata = nullptr;
			dims[0] = 0;
			dims[1] = 0;
		}
	}
	

	cuMatrix& operator =(cuMatrix&& temp) { std::swap(pdata, temp.pdata); std::swap(dims, temp.dims); return *this; }



	void resize(int m, int n, bool copyData = false) {
		if (m == dims[0] && n == dims[1]) return;
		cuMatrix<R> m2(m, n);
		if (copyData && dims[0] * dims[1] > 0) checkCudaErrors(cudaMemcpyAsync(m2.pdata, pdata, std::min(int(dims[0] * dims[1]), m * n) * sizeof(R), cudaMemcpyDeviceToDevice));
		std::swap(*this, m2);
	}

	void reshape(int m, int n)
	{
		if (m*n != dims[0] * dims[1])return;
		if (m * n != dims[0] * dims[1])return;
		dims[0] = m;
		dims[1] = n;
	}

	void LUSolver_init()
	{
		if (solver) delete solver;
		int m = dims[0];
		int n = dims[1];
		if (m != n) { printf("not square matrix"); return; }
		solver = new cuLUSolverDn<R>;
		solver->init(n);

	}

	void CholSolver_init()
	{
		if (solver) delete solver;
		int m = dims[0];
		int n = dims[1];
		if (m != n) { printf("not square matrix"); return; }
		solver = new cuCholSolverDn<R>;
		solver->init(n);
	}

	void Factor()
	{
		solver->factor(pdata);
	}

	void Solve(cuVector<R>& b)
	{
		if (b.len != dims[0])return;
		solver->solve(b.pdata);
	}

	void Solve(R* data)
	{
		solver->solve(data);
	}

};

#undef ChooseCusolverDnFunc 



/*
if T is a 3D tensor,[a,b,c]=size(T),

then T(i,j,k)=T.pdata[k*a*b+j*a+i]

*/
/*template<class R = real>
struct cuTensor3D
{
	R* pdata=nullptr;
	size_t dims[3];
	cuTensor3D(int a = 0, int b = 0, int c = 0)
	{
		dims[0] = a;
		dims[1] = b;
		dims[2] = c;
		if (a*b*c > 0)
		{
			checkCudaErrors(cudaMalloc((void**)&pdata, len * sizeof(R)));
		}
	}
	~cuTensor3D()
	{
		if(pdata){ checkCudaErrors(cudaFree(pdata));}
	}
	void Permutations132

};*/

/*
template<class R=real>
struct cuArray
{
	R** plist=nullptr;
	size_t len=0;
	cuArray() {}
	~cuArray() { if (plist) checkCudaErrors(cudaFree(plist)); }
	cuArray(R* data, int k, int n)
	{
		if (n > 0)
		{
			R** hostList = nullptr;
			len = n;
			hostList = (R**)malloc((sizeof(R*))*n);
			for (int i = 0; i < n; i++)
			{
				hostList[i] = data + i * k;
			}
			cudaMalloc((void**)&plist, (sizeof(R*))*n);
			cudaMemcpyAsync(plist, hostList, n * sizeof(R*), cudaMemcpyHostToDevice);
			free(hostList);
		}
	}
	//More function will be added in the future
};
*/
