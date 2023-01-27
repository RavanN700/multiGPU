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
 
// __global__ void delay(int *dest, int destDevice, int *src, int srcDevice,int num_elems,unsigned long long timeout_clocks = 10000) {                        
// // Wait until the application notifies us that it has completed queuing up the
// // experiment, or timeout and exit, allowing the application to make progress
// long long int start_clock, sample_clock;
// start_clock = clock64();
// cudaMemcpyPeer(dest, destDevice, src, srcDevice, sizeof(int) * num_elems);

// while (1) {
// sample_clock = clock64();

// if (sample_clock - start_clock > timeout_clocks) {
// break;
// }
// }
// }
 
 
int main(int argc, char **argv) {

    using namespace std;
    int numGPUs;

    int src;
    int det;
    int memsize;

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
    // Src GPU
    cudaSetDevice(src);
    cudaMalloc(&buffers[src], memsize * sizeof(int));
    cudaMemset(buffers[src], src, memsize * sizeof(int)); // Set buffer[src] to value src
    // Det GPU
    cudaSetDevice(det);
    cudaMalloc(&buffers[det], memsize * sizeof(int));
    cudaMemset(buffers[det], det, memsize * sizeof(int)); // Set buffer[det] to value det


    // Start profiler // nvprof --profile-from-start off
    cudaProfilerStart(); 

    // Copy data from src GPU to det GPU
    cudaSetDevice(src);
    // delay<<<128, 128>>>(buffers[det], det, buffers[src], src, sizeof(int) * memsize, 10000);
    cudaMemcpyPeer(buffers[det], det, buffers[src], src, sizeof(int) * memsize);
    

    // Stop profiler
    cudaProfilerStop(); 

    double mb = memsize * sizeof(int) / (double)1e6;
    printf("Size of data transfer (MB): %f\n", mb);


    exit(EXIT_SUCCESS);
 }
 