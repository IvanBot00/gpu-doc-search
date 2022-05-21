#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "support.h"
#include "kernel.cu"

int main (int argc, char *argv[])
{
    cudaError_t cuda_ret;

    srand(time(NULL));

    // Allocate host variables
    unsigned *tf_h, *tf_d;
    float *scores_h, *scores_d;
    size_t tf_sz, scores_sz;
    unsigned numDocs, numWords;

    if (argc == 1) {
	numDocs = 128;
	numWords = 1000;
    } else if (argc == 2) {
	numDocs = atoi(argv[2]);
	numWords = atoi(argv[2]);
    } else if (argc == 3) {
	numWords = atoi(argv[2]);
	numDocs = atoi(argv[3]);
    } else if (argc == 4) {
	printf("File read not implemented yet\n");
    } else {
	printf("\n    Invalid input parameters!");
    }
    
    tf_sz = numDocs * numWords;
    scores_sz = numDocs;

    tf_h = (unsigned*) malloc(sizeof(unsigned) * tf_sz);
    for (unsigned int i=0; i < tf_sz; ++i) { tf_h[i] = rand() % 10; }

    scores_h = (float*) malloc(sizeof(float) * scores_sz);
    for (unsigned int i=0; i < scores_sz; ++i) { scores_h[i] = 0; }
	
    // Allocate device variables
    printf("Allocating device variables..."); fflush(stdout);
    
    cudaMalloc((void**) &tf_d, sizeof(unsigned) * tf_sz);
    cudaMalloc((void**) &scores_d, sizeof(float) * scores_sz);

    cudaDeviceSynchronize();

    // Copy host variables to device
    printf("Copying data from host to device..."); fflush(stdout);

    cudaMemcpy(tf_d, tf_h, tf_sz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(scores_d, scores_h, scores_sz * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Launch Kernels ----------

    calculateBM25(scores_d, tf_d, numWords, numDocs);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    // -------------------------
    
    // Copy device variables to host

    cudaMemcpy(scores_h, scores_d, scores_sz * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Verify results

    printf("Verifying Results..."); fflush(stdout);

    verify_bm25(scores_h, tf_h, numWords, numDocs);

    // Free memory
    free(tf_h);
    free(scores_h);

    cudaFree(tf_d);
    cudaFree(scores_h);
}

