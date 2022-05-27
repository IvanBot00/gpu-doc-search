#include <stdio.h>
#include <math.h>
#include <cmath>
#include "support.h"
#include "defines.h"

bool isEqual(float a, float b, float error = 0.01f) {
    return fabs(a - b) < error;
}

void verify_df(const unsigned *gpuResult, const unsigned *tf, unsigned int numWords, unsigned int numDocs) {
    unsigned *cpuResult;
    cpuResult = (unsigned*)malloc(numWords * sizeof(unsigned));
    for (unsigned i=0; i < numWords; ++i) { cpuResult[i] = 0; }

    for (unsigned Row = 0; Row < numWords; ++Row) {
	for (unsigned Col = 0; Col < numDocs; ++Col) {
	    if (tf[Row * numDocs + Col] > 0)
		cpuResult[Row] += 1;
	}
    }


    for (unsigned i = 0; i < numWords; ++i) {
	printf("CPU: %u, GPU: %u\n", cpuResult[i], gpuResult[i]);
	if (gpuResult[i] != cpuResult[i]) {
	    printf("\nTEST FAILED %d\n", i);
	    printf("CPU: %u, GPU: %u\n", cpuResult[i], gpuResult[i]);
	    exit(1);
	}
    }

    printf("\n DF TEST PASSED\n");

    free(cpuResult);
}

void verify_bm25(const float *gpuResult, const unsigned *tf, unsigned int numWords, unsigned int numDocs) {

    float *cpuResult;
    cpuResult = (float*)malloc(numDocs * sizeof(float));
    for (unsigned i = 0; i < numDocs; ++i) { cpuResult[i] = 0.0; }

    for (unsigned Row = 0; Row < numWords; ++Row) {
	for (unsigned Col = 0; Col < numDocs; ++Col) {
 	    float doctf = tf[Row * numDocs + Col];
	    cpuResult[Col] += log((numDocs - 10 + 0.5) / (10 + 0.5)) * (2.2 * doctf / (K + doctf));    
	}
    }

    for (unsigned i = 0; i < numDocs; ++i) {
	if (!isEqual(cpuResult[i], gpuResult[i])) {
	    printf("\nTEST FAILED %d\n", i);
	    printf("CPU: %f, GPU: %f\n", cpuResult[i], gpuResult[i]);
	    exit(1);
	}
    }

    printf("\nTEST PASSED\n");
    
    free(cpuResult);
}
