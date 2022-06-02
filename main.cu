#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "support.h"
#include "kernel.cu"

int main (int argc, char *argv[])
{
    Timer timer;

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    cudaError_t cuda_ret;

    srand(time(NULL));

    // Allocate host variables
    unsigned *tf_h, *tf_d, *df_h, *df_d;
    float *scores_h, *scores_d;
    size_t tf_sz, scores_sz, df_sz;
    unsigned numDocs, numWords;

    if (argc == 1) {
	numDocs = 10;
	numWords = 10;
    } else if (argc == 2) {
	numDocs = atoi(argv[1]);
	numWords = atoi(argv[1]);
	printf("%u", numDocs);
    } else if (argc == 3) {
	numWords = atoi(argv[1]);
	numDocs = atoi(argv[2]);
    } else if (argc == 4) {
	printf("File read not implemented yet\n");
    } else {
	printf("\n    Invalid input parameters!");
    }
    
    tf_sz = numDocs * numWords;
    scores_sz = numDocs;
    df_sz = numWords;

    tf_h = (unsigned*) malloc(sizeof(unsigned) * tf_sz);
    for (unsigned int i=0; i < tf_sz; ++i) {
	unsigned initialVal = rand() % 2;
	tf_h[i] = initialVal * rand() % 9;
    }

    df_h = (unsigned*) malloc(sizeof(unsigned) * df_sz);
    for (unsigned int i=0; i < df_sz; ++i) { df_h[i] = 0; }

    scores_h = (float*) malloc(sizeof(float) * scores_sz);
    for (unsigned int i=0; i < scores_sz; ++i) { scores_h[i] = 0; }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	
    // Allocate device variables
    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);
    
    cudaMalloc((void**) &tf_d, sizeof(unsigned) * tf_sz);
    cudaMalloc((void**) &scores_d, sizeof(float) * scores_sz);
    cudaMalloc((void**) &df_d, sizeof(unsigned) * df_sz);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(tf_d, tf_h, tf_sz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemset(df_d, 0, numWords * sizeof(unsigned));
    cudaMemcpy(scores_d, scores_h, scores_sz * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch Kernels ----------
    
    printf("Launching kernels..."); fflush(stdout);
    startTime(&timer);

    //calculateDocFrequency(df_d, tf_d, tf_h, numWords, numDocs);
    calculateDocFrequency(df_d, tf_d, numWords, numDocs);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    calculateBM25(scores_d, tf_d, df_d, numWords, numDocs);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // -------------------------
    // Copy device variables to host

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(scores_h, scores_d, scores_sz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(df_h, df_d, df_sz * sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify results

    printf("Verifying Results..."); fflush(stdout);

    startTime(&timer);

    verify_df(df_h, tf_h, numWords, numDocs);
    verify_bm25(scores_h, tf_h, df_h, numWords, numDocs);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Free memory
    free(tf_h);
    free(scores_h);
    free(df_h);

    cudaFree(tf_d);
    cudaFree(scores_d);
    cudaFree(df_d);
}

