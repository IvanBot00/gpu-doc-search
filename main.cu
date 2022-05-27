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
    unsigned *tf_h, *tf_d, *df_h, *df_d;
    float *scores_h, *scores_d;
    size_t tf_sz, scores_sz, df_sz;
    unsigned numDocs, numWords;

    if (argc == 1) {
	numDocs = 1000;
	numWords = 1000;
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

	
    // Allocate device variables
    printf("Allocating device variables..."); fflush(stdout);
    
    cudaMalloc((void**) &tf_d, sizeof(unsigned) * tf_sz);
    cudaMalloc((void**) &scores_d, sizeof(float) * scores_sz);
    cudaMalloc((void**) &df_d, sizeof(unsigned) * df_sz);

    cudaDeviceSynchronize();

    // Copy host variables to device
    printf("Copying data from host to device..."); fflush(stdout);

   // cudaMemcpy(tf_d, tf_h, tf_sz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemset(df_d, 0, numWords * sizeof(unsigned));
    cudaMemcpy(scores_d, scores_h, scores_sz * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Launch Kernels ----------

    calculateDocFrequency(df_d, tf_d, tf_h, numWords, numDocs);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    calculateBM25(scores_d, tf_d, df_d, numWords, numDocs);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    // -------------------------
    
    // Copy device variables to host

    cudaMemcpy(scores_h, scores_d, scores_sz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(df_h, df_d, df_sz * sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Verify results

    printf("Verifying Results..."); fflush(stdout);

    verify_df(df_h, tf_h, numWords, numDocs);
    verify_bm25(scores_h, tf_h, df_h, numWords, numDocs);


    // Free memory
    free(tf_h);
    free(scores_h);
    free(df_h);

    cudaFree(tf_d);
    cudaFree(scores_d);
    cudaFree(df_d);
}

