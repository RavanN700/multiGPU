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
 
__global__ void delay(volatile int *flag,
    unsigned long long timeout_clocks = 1000000000) {                        
// Wait until the application notifies us that it has completed queuing up the
// experiment, or timeout and exit, allowing the application to make progress
long long int start_clock, sample_clock;
start_clock = clock64();

while (!*flag) {
sample_clock = clock64();

if (sample_clock - start_clock > timeout_clocks) {
break;
}
}
}
 
 
int main(int argc, char **argv) {

    using namespace std;
    int numGPUs;

    int src;
    int det;
    int memsize;
    volatile int *flag = NULL;

    cudaGetDeviceCount(&numGPUs); // get number of GPUs

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
    vector<int *> buffers(numGPUs);
    // vector<cudaStream_t> stream(numGPUs);
    // Src GPU
    cudaSetDevice(src);
    // cudaStreamCreateWithFlags(&stream[src], cudaStreamNonBlocking);
    cudaMalloc(&buffers[src], memsize * sizeof(int));
    cudaMemset(buffers[src], src, memsize * sizeof(int)); // Set buffer[src] to value src
    // Det GPU
    cudaSetDevice(det);
    // cudaStreamCreateWithFlags(&stream[det], cudaStreamNonBlocking);
    cudaMalloc(&buffers[det], memsize * sizeof(int));
    cudaMemset(buffers[det], det, memsize * sizeof(int)); // Set buffer[det] to value det

    // Start profiler // nvprof --profile-from-start off
    cudaProfilerStart(); 

    // Copy data from src GPU to det GPU
    // cudaSetDevice(src);
    // cudaStreamSynchronize(stream[src]);
    // cudaMemcpyPeerAsync(buffers[det], det, buffers[src], src, sizeof(int) * memsize, stream[src]);
    *flag = 0;
    delay<<<128, 128, 0, stream[i]>>>(flag);
    cudaMemcpyPeer(buffers[det], det, buffers[src], src, sizeof(int) * memsize);
    // cudaStreamSynchronize(stream[src]);
    
    // Stop profiler
    // cudaProfilerStop(); 

    double mb = memsize * sizeof(int) / (double)1e6;
    printf("Size of data transfer (MB): %f\n", mb);


    exit(EXIT_SUCCESS);
 }
 