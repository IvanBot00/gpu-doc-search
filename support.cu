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

/*
    for (unsigned i = 0; i < numWords; ++i) {
	for (unsigned j = 0; j < numDocs; ++j) {
	    printf(" %u ", tf[i * numWords + j]);
        }
        printf("\n");
    }
*/

    for (unsigned i = 0; i < numWords; ++i) {
	//printf("CPU: %u, GPU: %u\n", cpuResult[i], gpuResult[i]);
	if (gpuResult[i] != cpuResult[i]) {
	    printf("\nDF TEST FAILED %d\n", i);
	    printf("CPU: %u, GPU: %u\n", cpuResult[i], gpuResult[i]);
	    exit(1);
	}
    }

    //printf("\nDF TEST PASSED\n");

    free(cpuResult);
}

void verify_bm25(const float *gpuResult, const unsigned *tf, const unsigned *df, unsigned int numWords, unsigned int numDocs) {

    float *cpuResult;
    cpuResult = (float*)malloc(numDocs * sizeof(float));
    for (unsigned i = 0; i < numDocs; ++i) { cpuResult[i] = 0.0; }

    unsigned doctf, docf;

    for (unsigned Row = 0; Row < numWords; ++Row) {
	for (unsigned Col = 0; Col < numDocs; ++Col) {
            doctf = tf[Row * numDocs + Col];
	    docf = df[Row];
	    cpuResult[Col] += log((numDocs - docf + 0.5) / (docf + 0.5)) * (2.2 * doctf / (K + doctf)); 
	}
    }

    for (unsigned i = 0; i < numDocs; ++i) {
	if (!isEqual(cpuResult[i], gpuResult[i])) {
	    printf("\nBM25 TEST FAILED %d\n", i);
	    printf("CPU: %f, GPU: %f\n", cpuResult[i], gpuResult[i]);
	    exit(1);
	}
    }

    //printf("\nBM25 TEST PASSED\n");
    
    free(cpuResult);
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
		+ (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
