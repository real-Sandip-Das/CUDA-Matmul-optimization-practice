#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CUDA_CALL_VOID(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CEIL_DIV(X, Y) (int)ceil(X/Y)

void initialize(int& M, int& N, int& K, float* &h_A, float* &h_B, float* &h_C_init, float* &h_C_final, float* &d_A, float* &d_B, float* &d_C1, float* &d_C2)
{
  // Input from the User
  std::cout << "Enter M, N and K:" << std::endl;
  std::cin >> M >> N >> K;
  // M = 2, N = 2, K = 2; // Debug
  h_A = (float *)malloc(M*K*sizeof(float));
  h_B = (float *)malloc(K*N*sizeof(float));
  h_C_init = (float *)malloc(M*N*sizeof(float));
  h_C_final = (float *)malloc(M*N*sizeof(float));

  // Allocate Device Memory
  CUDA_CALL_VOID(cudaMalloc((void **)&d_A, M*K*sizeof(float)));
  CUDA_CALL_VOID(cudaMalloc((void **)&d_B, K*N*sizeof(float)));
  CUDA_CALL_VOID(cudaMalloc((void **)&d_C1, M*N*sizeof(float)));
  CUDA_CALL_VOID(cudaMalloc((void **)&d_C2, M*N*sizeof(float)));

  // Randomly generate matrices A, B, C
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateUniform(prng, d_A, M*K);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateUniform(prng, d_B, K*N);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateUniform(prng, d_C1, M*N);
  cudaMemcpy(d_C2, d_C1, M*N*sizeof(float), cudaMemcpyDeviceToDevice);
  curandDestroyGenerator(prng);
  
  // Copy generated Matrices to host
  CUDA_CALL_VOID(cudaMemcpy(h_A, d_A, M*K*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL_VOID(cudaMemcpy(h_B, d_B, K*N*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL_VOID(cudaMemcpy(h_C_init, d_C1, M*N*sizeof(float), cudaMemcpyDeviceToHost));
}

__global__ void match_results(float *A, float *B, int* flags, int N, float eps)
{
  const uint BLOCKSIZE = blockDim.x;
  const uint i = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint j = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  const uint index = i * N + j;
  flags[index] = (abs(A[index] - B[index]) < eps);
}

void check_all(int *flags, int len)
{
  bool res = true;
  for (int i = 0; i < len; ++i) {
      res = res && flags[i];
  }
  if (res) std::cout << "Matrix C calculated by cuBLAS matches with that of the implemented kernel.\n";
  else std::cout << "Warning: Results not matched!\n";
}

void print_matrix(int height, int width, float *Mat) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << Mat[i*width + j] << ' ';
    }
    std::cout << '\n';
  }
}

void print_result(int M, int N, int K, float* h_A, float* h_B, float* h_C_init, float* h_C_final)
{
  //Printing h_A
  std::cout << "Matrix A:\n";
  print_matrix(M, K, h_A);

  //Printing h_B
  std::cout << "Matrix B:\n";
  print_matrix(K, N, h_B);

  //Printing h_C_init
  std::cout << "Initial matrix C:\n";
  print_matrix(M, N, h_C_init);

  //Printing h_C_final
  std::cout << "Final matrix C:\n";
  print_matrix(M, N, h_C_final);
}

__global__ void kernel_1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < N) {
    float acc = 0.0;
    for (int k = 0; k < K; ++k) {
      acc += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = alpha * acc + beta * C[i * N + j];
  }
}

__global__ void kernel_2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint BLOCKSIZE = 32;
  const uint i = blockIdx.x * blockDim.x + (threadIdx.x / BLOCKSIZE);
  const uint j = blockIdx.y * blockDim.y + (threadIdx.x % BLOCKSIZE);

  if (i < M && j < N) {
    float acc = 0.0;
    for (int k = 0; k < K; ++k) {
      acc += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = alpha * acc + beta * C[i * N + j];
  }
}

__global__ void kernel_3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint BLOCKSIZE = 32;
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.y / BLOCKSIZE;

  float acc = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      acc += As[threadRow * BLOCKSIZE + dotIdx] *
              Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * acc + beta * C[threadRow * N + threadCol];
}

int main()
{
  int M, N, K;
  float *h_A, *h_B, *h_C_init, *h_C_final;
  float *d_A, *d_B, *d_C_kernel, *d_C_cuBLAS;
  float alpha = 1.0f, beta = 0.1f;
  initialize(M, N, K, h_A, h_B, h_C_init, h_C_final, d_A, d_B, d_C_kernel, d_C_cuBLAS);
  // TODO: generate alpha and beta randomly too

  // Execute implemented kernel and measure runtime
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  // dim3 blockDim(32, 32, 1); // Kernel 1
  dim3 blockDim(32*32, 1); // Kernel 2, 3
  cudaEvent_t start_implem, stop_implem;
  cudaEventCreate(&start_implem);
  cudaEventCreate(&stop_implem);
  cudaEventRecord(start_implem);
  // kernel_1<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C_kernel);
  // kernel_2<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C_kernel);
  kernel_3<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C_kernel);
  cudaDeviceSynchronize();
  cudaEventRecord(stop_implem);
  cudaEventSynchronize(stop_implem);

  // Execute cuBLAS kernel and measure runtime
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaEvent_t start_cuBLAS, stop_cuBLAS;
  cudaEventCreate(&start_cuBLAS);
  cudaEventCreate(&stop_cuBLAS);
  cudaEventRecord(start_cuBLAS);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C_cuBLAS, M);
  cudaEventRecord(stop_cuBLAS);
  cudaEventSynchronize(stop_cuBLAS);
  cublasDestroy(handle);

  // Match the results of both of the kernels
  int h_flags[M*N];
  int* d_flags;
  CUDA_CALL(cudaMalloc((void **)&d_flags, M*N*sizeof(int)));
  match_results<<<gridDim, blockDim>>>(d_C_kernel, d_C_cuBLAS, d_flags, N, 0.001);
  CUDA_CALL(cudaMemcpy(h_flags, d_flags, M*N*sizeof(int) ,cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  check_all(h_flags, M*N);
  // TODO: Parallelize matching

  // Copy resultant matrix to Host
  cudaMemcpy(h_C_final, d_C_kernel, M*N*sizeof(float), cudaMemcpyDeviceToHost);

  // Print the results
  print_result(M, N, K, h_A, h_B, h_C_init, h_C_final);

  // Speed Comparison
  float milliseconds_implem, milliseconds_cuBLAS;
  cudaEventElapsedTime(&milliseconds_implem, start_implem, stop_implem);
  cudaEventElapsedTime(&milliseconds_cuBLAS, start_cuBLAS, stop_cuBLAS);
  cudaEventDestroy(start_implem);
  cudaEventDestroy(stop_implem);
  cudaEventDestroy(start_cuBLAS);
  cudaEventDestroy(stop_cuBLAS);
  std::cout << "Elapsed Time(Implemented kernel): " << milliseconds_implem << " ms\n";
  std::cout << "Elapsed Time(cuBLAS kernel): " << milliseconds_cuBLAS << " ms\n";

  // Freeing allocated Memory
  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C_kernel));
  CUDA_CALL(cudaFree(d_C_cuBLAS));
  free(h_A);
  free(h_B);
  free(h_C_init);
  free(h_C_final);
}