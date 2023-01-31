#include <vector>
#include <cuda_profiler_api.h> // For cudaProfilerStart() and cudaProfilerStop()
#include <helper_cuda.h>
#include <helper_timer.h>
#include <cstdio>
#include <string>
#include <thrust/device_vector.h>
#include <fstream>
#include <cupti_profiler.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
 
 // Vector kernel 
__global__ void
vecAdd(const int *A, const int *B, int *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

// initilize vector to be all "1"
static void
initVec(int *vec, int n)
{
    for (int i = 0; i < n; i++)
        vec[i] = 1;
}

 
int main(int argc, char **argv) {

    using namespace std;
    // int numGPUs;

    int src;
    int det;
    int size;
    int memsize;

    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;


    // cudaGetDeviceCount(&numGPUs); // get number of GPUs

    printf("Please enter the source GPU: ");
    scanf("%d", &src);
    printf("\n");
    printf("Please enter the detination GPU: "); 
    scanf("%d", &det);
    printf("\n");
    printf("Please enter the Number of Elements: ") ;
    scanf("%d", &memsize);
    printf("\n");

    size_t size = memsize * sizeof(int);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize input vectors
    initVec(h_A, 1);
    initVec(h_B, 2);
    memset(h_C, 0, size);

    // Allocate vectors on Src and Det GPU

    // Src GPU contains vec_A and vec_C
    cudaSetDevice(src);
    cudaMalloc((void**)&d_A, size);  
    cudaMalloc((void**)&d_C, size);

    // Copy vector A from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Det GPU
    cudaSetDevice(det);
    cudaMalloc(&buffers[det], size * sizeof(int));

    int deviceList[8] = {0,1,2,3,4,5,6,7};
    cudaSetValidDevices(deviceList, 2);

    // Copy vector B from host memory to device memory
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    


    int threadsPerBlock = 256;
    int blocksPerGrid = (memsize + threadsPerBlock - 1) / threadsPerBlock;
    
    // Start profiler // nvprof --profile-from-start off
    cudaProfilerStart(); 
    

    vecAdd <<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, memsize);

    

    // Stop profiler
    cudaProfilerStop(); 


    // Copy back to host memory in src GPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);




    double mb = memsize * sizeof(int) / (double)1e6;
    printf("Size of data transfer (MB): %f\n", mb);

    printf("Output vector: %d\n", h_C[0]);
    
    cudaFree(buffers[src]);
    cudaFree(buffers[det]);
    cudaFree(buffers[9]);
    free(h_A);
    free(h_B);
    free(h_C);

    exit(EXIT_SUCCESS);
 }
 