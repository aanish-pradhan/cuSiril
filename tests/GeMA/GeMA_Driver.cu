// HEADER FILES
#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	// Command line arguments
	if (argc < 3)
	{
		printf("Usage: ./gema [Nunber of Matrices] [Dimension]");
	}
	int numberOfMatrices = atoi(argv[1]);
	int dimension = atoi(argv[2]);
	

	// Allocate matrices/tensors on HOST
	float *h_A, *h_B;
	h_A = (float*) malloc((dimension * dimension * numberOfMatrices) * 
		sizeof(float)); // Input
	h_B = (float*) malloc((dimension * dimension) * sizeof(float)); // Output

	// Populate input tensor
	
	/*
	For reproducability, every element is 1.0. That way, solutions are easily 
	computable by hand.
	*/
	
	double tic = omp_get_wtime();
	for (int i = 0; i < dimension * dimension * numberOfMatrices; i++)
	{
		h_A[i] = 1.0;
	}
	double toc = omp_get_wtime();
	printf("Population time (s): %g\n", toc - tic);

	// Allocate matrices/tensors on DEVICE
	float *d_A, *d_B;
	cudaMalloc(&d_A, (dimension * dimension * numberOfMatrices) * 
		sizeof(float));
	cudaMalloc(&d_B, (dimension * dimension) * sizeof(float));

	// Copy data to DEVICE
	cudaEvent_t d_tic, d_toc;
	cudaEventCreate(&d_tic);
	cudaEventCreate(&d_toc);
	cudaEventRecord(d_tic);
	cudaMemcpy(d_A, h_A, (dimension * dimension * numberOfMatrices) * 
		sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(d_toc);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d_tic, d_toc);
	printf("Data transfer time (s): %g\n", elapsedTime / 1000.0);

	cudaMemcpy(d_B, h_B, (dimension * dimension) * sizeof(float), 
		cudaMemcpyHostToDevice);

	// Launch kernel

	// TODO

	// Free arrays
	free(h_A);
	free(h_B);
	cudaFree(d_A);
	cudaFree(d_B);

	return 0;
}

