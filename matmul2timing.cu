// nvcc 001 isamax .c -lcublas
#include <iostream>
#include </usr/include/stdio.h>
#include </usr/include/stdlib.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "cublas_v2.h"
#include "curand.h"
#include "cuda_fp16.h"
#include <math.h>
#include <time.h>
#include <library_types.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>
#include "common.h"

using namespace std;


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = (in[idx]);
   }
}
__global__ void convertFp16ToFp32 (float *out, half *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = (in[idx]);
   }
}
/*
__global__ void convertFp32ToFp16 (__half *out, float *in, int rows, int cols) {
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){   
				out[i * cols + j] = __float2half(in[i * cols + j]);
			}
		}
}
*/
 void print_matrix(float *A, int nr_rows_A, int nr_cols_A) { 
     for(int i = 0; i < nr_rows_A; i++){
         for(int j = 0; j < nr_cols_A; j++){
             std::cout << A[i * nr_cols_A + j] << " ";
         }
         std::cout << std::endl;
     }
     std::cout << std::endl;
 }

// Fill the array with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

	
void gpu_blas_mmul(__half *A, __half *B, __half *C, int m, int k, int n) {
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
		cublasStatus_t cublasStat  = cublasCreate(&handle);
		// Set the math mode to allow cuBLAS to use Tensor Cores:
		cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
		//cublasStat = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		

		//n maps to the output dimension.
		//m is the batch size * seq length.
		//k maps to the input dimension.

		//leading dimension of B will be cols in B(host) and it will be accessed as T.
		//leading dimension of A will be cols in A(host) and it will be accessed as N.
		//Leading dimension of C will be cols in C(host) and it will be accesses as N. 
			
		//A is m * k in host k * m in device.
		//B is n * K in host k * n in device. 
		//C is m * n in host n * m in device. 

		//m will be rows A, C.
		//k will be cols A, B.
		//n will be rows B, cols in C.
		int lda = k, ldb = k, ldc = n;
	//-------------------------------peforming warmup runs-------------------------------------//
		for(int i = 0; i < 500; i++){
			// Do the actual multiplication
	check_cuda_error(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alpha, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, beta, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
	//-------------------------------------perform actual runs--------------------------------//
		cudaDeviceSynchronize();
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
		int niter = 10000;
		for(int i = 0; i < niter; i++){
			// Do the actual multiplication
		//	cublasStat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alpha, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, beta, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	check_cuda_error(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alpha, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, beta, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		//cout<<cublasStat<<endl;
			//cublasStat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1024, 1536, 4096, alpha, B, CUDA_R_16F, 1024, A, CUDA_R_16F, 4096, beta, C,CUDA_R_16F,1024 ,CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

		cudaEventRecord(stop, NULL);

		//stop event to complete
		cudaEventSynchronize(stop);

		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		cout<<"Total time Taken: "<<msecTotal<<" msec"<<endl;

		// Compute and print the performance
		float msecPerMatrixMul = msecTotal/niter;
		cout<<"Average time taken per matmul: "<<msecPerMatrixMul<<" msec"<<endl;
		
		double flopsPerMatrixMul = 2.0 * (double) m * (double) n * (double) k;
		double teraFlops = (flopsPerMatrixMul * 1.0e-12f) / (msecPerMatrixMul / 1000.0f);
		printf(
				"Performance= %.2f TFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
				teraFlops,
				msecPerMatrixMul,
				flopsPerMatrixMul);
/*
		for(int i = 0; i < 20; i++){
			cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, CUDA_R_16F, lda, 384 * 384, B, CUDA_R_16F, ldb, 384 * 64, beta, C, CUDA_R_16F, ldc, 384 * 64, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
		}
*/
    // Destroy the handle
    cublasDestroy(handle);
}

int main() {
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    
		//A is for the Activations. has dimensions m * k, where m is (seq length * batchsize),  k is no of inputs to the layer. 
		//B is for the weights. stored as B' at host. has dimensions n * k. n is the number of outputs k is the not of inputs to the layer.  
		//C is the output matrix. has dimensions m * n.
		//Matmul will be A B'.
		//set dims according to operation c = a * b'

		nr_rows_A = 4096;
		nr_cols_A = 4096;
		nr_rows_B = 4096;
		nr_cols_B = 4096;
		nr_rows_C = 4096;
		nr_cols_C = 4096;

    // Allocate 6 arrays on GPU.
		// array on device of type half.
		// float because curand generates only fp32 numbers.
		// __half arrays for fp16 numbers.
		float *df_A, *df_B, *df_C;	
    __half *d_A, *d_B, *d_C;


    check_cuda_error(cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(__half)));
    check_cuda_error(cudaMalloc(&df_A,nr_rows_A * nr_cols_A * sizeof(float)));
		GPU_fill_rand(df_A, nr_rows_A, nr_cols_A);	
		convertFp32ToFp16 <<< (nr_rows_A * nr_cols_A+ 255) / 256, 256 >>> (d_A, df_A, nr_rows_A * nr_cols_A);

 
		check_cuda_error(cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(__half)));
    check_cuda_error(cudaMalloc(&df_B,nr_rows_B * nr_cols_B * sizeof(float)));
		GPU_fill_rand(df_B, nr_rows_B, nr_cols_B);	
		convertFp32ToFp16 <<< (nr_rows_B * nr_cols_B + 255) / 256, 256 >>> (d_B, df_B, nr_rows_B * nr_cols_B);
    
		check_cuda_error(cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(__half)));
    check_cuda_error(cudaMalloc(&df_C,nr_rows_C * nr_cols_C * sizeof(float)));
 
			
		
  
		//m will be rows a.
		//k will be cols a.
		//n will be rows b.
		//call the matmul function.
	  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_rows_B);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(df_A);
    cudaFree(df_B);
    cudaFree(df_C);


    return 0;
}

