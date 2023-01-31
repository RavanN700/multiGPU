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

 
int main(int argc, char **argv) {

    using namespace std;
    // int numGPUs;

    int src;
    int det;
    int memsize;

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


    // Initilize buffer for src and det GPU
    vector<int *> buffers(10);
    // Src GPU
    cudaSetDevice(src);
    cudaMalloc(&buffers[src], memsize * sizeof(int));     // vec A  
    cudaMemset(buffers[src], src, memsize * sizeof(int)); // Set buffer[src] to value src

    cudaMalloc(&buffers[9], memsize * sizeof(int)); // vec C
    cudaMemset(buffers[9], 9, memsize * sizeof(int));

    // Det GPU
    cudaSetDevice(det);
    cudaMalloc(&buffers[det], memsize * sizeof(int));
    cudaMemset(buffers[det], det, memsize * sizeof(int)); // Set buffer[det] to value det

    cudaSetDevice(src);
    cudaSetDevice(det);
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (memsize + threadsPerBlock - 1) / threadsPerBlock;
    // Start profiler // nvprof --profile-from-start off
    cudaProfilerStart(); 
    

    vecAdd <<<256, 256>>>(buffers[src], buffers[det], buffers[9], memsize);

    // Copy data from src GPU to det GPU
    // cudaMemcpyPeer(buffers[det], det, buffers[src], src, sizeof(int) * memsize);
    

    // Stop profiler
    cudaProfilerStop(); 

    cudaFree(buffers[src]);
    cudaFree(buffers[det]);
    cudaFree(buffers[9]);


    double mb = memsize * sizeof(int) / (double)1e6;
    printf("Size of data transfer (MB): %f\n", mb);


    exit(EXIT_SUCCESS);
 }
 