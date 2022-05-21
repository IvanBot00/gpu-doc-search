#include <stdio.h>
#include <math.h>
#include "support.h"
#include "defines.h"

void verify_bm25(const float *gpuResult, const unsigned *tf, unsigned int numWords, unsigned int numDocs) {

    const float relativeTolerance = 1e-10;

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
        float relativeError = cpuResult[i] - gpuResult[i] / cpuResult[i];

	if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
	    printf("\nTEST FAILED %d\n", i);
	    printf("CPU: %f, GPU: %f\n", cpuResult[i], gpuResult[i]);
	    exit(1);
	}
    }

    printf("\nTEST PASSED\n");
}
