#include <stdio.h>
#include "defines.h"

/*
__device__ float magnitude() {

}

__global__ void vectorSpaceKernel() {
    // calculate cosine (tf[row][col] * query[row]

   // __syncthreads;

   // calculate magnitude of query

   // calculate magnitude of doc horizontal memory access :( (opportunity for dynamic paralleism)

    // calculate product of magntidues

    // divide result by product
}

void calculateVectorSpace() {

}
*/
/*
__device__ int docFrequency() {

}
*/
__global__ void bm25Kernel(float *output, const unsigned *tf, const unsigned numWords, const unsigned numDocs) {

    unsigned Col = blockDim.x * blockIdx.x + threadIdx.x;

    float docScore = 0.0;

    for (int Row = 0; Row < numWords; ++Row) {
	int df = 10; //docFrequency(tf, Row, numDocs);
	int doctf = tf[Row * numDocs + Col];
	docScore += logf((numDocs - df + 0.5) / (df + 0.5)) * ((K_1 + 1) * doctf / (K + doctf));	
    } 

    __syncthreads;

    if (Col < numDocs) {
	output[Col] = docScore;
    }
}

void calculateBM25(float *output, const unsigned *tf, const unsigned numWords, const unsigned numDocs) {

    int BLOCK_SIZE = 512;

    dim3 dimGrid((numDocs - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    bm25Kernel<<<dimGrid, dimBlock>>>(output, tf, numWords, numDocs);
}
/*
__device__ float dotProduct(float *a, float *b, int size) {

}

__device__ void cosineSimilarity() {

    // calculate magntidue

    // calc dot product

    // calc dot proudct / magntiude

    // set result to array index 

}

void scoreDocuments(const int *tf, const float *idf, const int numWords, const int numDocs) {

    dim3 dimGrid = 4;
    dim3 dimBlock = 512;

    for (unsigned int i = 0; i < num_docs; i += 2) {

	// stream = new cudastream
	// stream 2 = new cudastream

	// memcpy async

	
     }
}
*/
/*
__global__ void calculateIDF(int size, int d, int w, const float *m, const float *idf, const float *buffer) {

    int i = threadIdx.x + blockId.x * blockDim.x;

    int stride = blockDim.x * gridDim.x;

    while (i < size) {
	int tf = m[i];
	if (tf > 0) {
	    // atomic add to the buffer
	}
	i += stride;
    }

    if (threadIdx.x < w) {
	//atomic add idf buffer to idf array
    }
}

void launchIDF(int size, int d, int w, const float *m, const float *idf) {
    
    dim3 dimGrid();
    dim3 dimBlock();

    calculateIDF<<<dimGrid, dimBlock>>>(d, w, m, idf);
}
*/

/*
__global__ void calculateScore() {
    
    // dot product
    for (int i = 0; i < numWords; ++i) {
	result[i] += tfidf[i] * q[i]
    }   
}

void launchScore(int size, int numDocs, int numWords, const float *m, const float *q) {

    SegSize = 1024 * numWords;

    dim3 DimGrid = SegSize / 256;
    dim3 DimBlock = 256;

    // load query into global mem

    // Use cuda streams to calculate document score
    for (int i = 0; i < size; i += SegSize * 2) {
	cudaMemcpyAsync(m0_d, m + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(m1_d, m + i + SegSize, SegSize * sizeof(float), cudaMemCpyHostToDevice, stream1);

	calculateScore<<<DimGrid, DimBlock, 0, stream0>>>(m0_d, SegSize, numDocs, numWords);
	calculateScore<<<DimGrid, DimBlock, 0, stream1>>>(m1_d, SegSize, numDocs, numWords);

	cudaMemcpyAsync(results_h + i, results0_d, SegSize * sizeof(float), cudaMemCpyDeviceToHost, stream0);
	cudaMemcpyAsync(results_h + i + SegSize, results1_d, SegSize * sizeof(float), cudaMemCpyDeviceToHost, stream1);
    }
}
*/
