#include <stdio.h>
#include "defines.h"


__global__ void docFrequencyKernel(unsigned *output, unsigned *input, unsigned numDocs) {

    __shared__ unsigned private_df;

    int t = threadIdx.x;

    if (t == 0)
	private_df = 0;

    //__syncthreads;

    int i = t + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < numDocs) {
	if (input[i] > 0) // maybe check if within bounds?
	    atomicAdd(&private_df, 1);
	i += stride;
    }

   // __syncthreads;

    if (threadIdx.x == 0) {
	atomicAdd(output, private_df);
    }
}


void calculateDocFrequency(unsigned *df_d, unsigned *tf_d, const unsigned *tf_h, const unsigned numWords, const unsigned numDocs) {
    
    int BLOCK_SIZE = 256;
    
    dim3 dimGrid((numDocs - 1) / BLOCK_SIZE + 1, 1 , 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    size_t rowSize = numDocs * sizeof(unsigned);

    unsigned *doc0, *doc1;

    for (unsigned i = 0; i < numWords; i += 2) {
	doc0 = tf_d + (i * numDocs);
	doc1 = tf_d + ((i+1) * numDocs);

	cudaMemcpyAsync(doc0, tf_h + i * numDocs, rowSize, cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(doc1, tf_h + (i + 1) * numDocs, rowSize, cudaMemcpyHostToDevice, stream1);

	docFrequencyKernel<<<2, BLOCK_SIZE, 0, stream0>>>(&df_d[i], doc0, numDocs);
	docFrequencyKernel<<<2, BLOCK_SIZE, 0, stream1>>>(&df_d[i+1], doc1, numDocs);
    }
}

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
