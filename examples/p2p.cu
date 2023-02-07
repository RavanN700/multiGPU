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
initVec(int *vec, int n, int value)
{
    for (int i = 0; i < n; i++)
        vec[i] = value;
}

 
int main(int argc, char **argv) {

    using namespace std;
    // int numGPUs;

    int src;
    int det;
    int memsize;

    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    struct timeval t1, t2;



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
    initVec(h_A, memsize, 1);
    initVec(h_B, memsize, 2);
    initVec(h_C, memsize, 3);

    // Src GPU contains vec_A and vec_C
    cudaSetDevice(src);
    cudaMalloc((void**)&d_A, size);  
    cudaMalloc((void**)&d_C, size);

    // Copy vector A from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Det GPU contains vec_B
    cudaSetDevice(det);
    cudaMalloc((void**)&d_B, size);

    // Copy vector B from host memory to device memory
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Make src and det device both valid
    // int deviceList[4] = {0,1,2,3};
    // cudaSetValidDevices(deviceList, 2);
    // cudaSetDevice(src);
    int threadsPerBlock = 256;
    int blocksPerGrid = (memsize + threadsPerBlock - 1) / threadsPerBlock;
    
    // Start record time
    // cudaEventRecord(start);
    gettimeofday(&t1, 0);    
    

    // Start profiler // nvprof --profile-from-start off
    cudaProfilerStart(); 
    

    

    // vecadd kernel
    // vecAdd <<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, memsize);

    // Peer to peer memory copy from device src to device det
    cudaSetDevice(src);
    cudaMemcpyPeer(d_B, det, d_A, src, size);
    cudaSetDevice(det);
    cudaMemcpyPeer(d_B, det, d_A, src, size);

    
    // Stop profiler
    cudaProfilerStop(); 

    // Stop time record
    // cudaEventRecord(stop);
    gettimeofday(&t2, 0);

    // cudaEventSynchronize(stop);
    // double milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    double milliseconds = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;


    // Copy back to host memory in src GPU
    // cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost); // test peer2peer memcpy


    double mb = memsize * sizeof(int) / (double)1e6;
    printf("Size of data transfer (MB): %f\n", mb);
    printf("Vector V_A (original value = 1): %d\n",h_A[memsize-1]);
    printf("Vector V_B (original value = 2): %d\n",h_B[memsize-1]);
    // printf("Vector V_C[memsize-1] (original value = 3): %d\n", h_C[memsize-1]);
    printf("Time (ms): %f\n", milliseconds);
    printf("Bandwith (MB/s): %f\n",mb*1e3/milliseconds);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    exit(EXIT_SUCCESS);
 }
 