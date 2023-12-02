/**
 * Implements functions for image stacking.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-02 
 */

// INCLUDE LIBRARIES
#include <cuda.h>
extern "C"
{
	#include "stack.h"
	#include "stack.cuh"
}
#include <stdlib.h>

// FUNCTIONS
/**
 * Initializes a Stack with unstacked subframes.
 * 
 * @param numberOfSubframes Number of subframes in image stack
 * @param imageWidth Subframe image width (px)
 * @param imageHeight Subframe image height (px)
 * @param redSubframes Red channels of flattened subframes
 * @param greenSubframes Green channels of flattened subframes
 * @param blueSubframes Blue channels of flattened subframes
 * @return Pointer to a Stack on the HOST
 */
Stack* initializeStackHOST(uint64_t numberOfSubframes, uint64_t imageWidth, 
	uint64_t imageHeight, uint16_t* redSubframes, uint16_t* greenSubframes, 
		uint16_t* blueSubframes)
{
	// Allocate Stack on the HOST
	Stack* h_imageStack = (Stack*) malloc(sizeof(Stack));

	// Populate the Stack attributes
	h_imageStack->numberOfSubframes = numberOfSubframes;
	h_imageStack->imageWidth = imageWidth;
	h_imageStack->imageHeight = imageHeight;

	uint64_t pixelsPerImage = imageWidth * imageHeight;
	h_imageStack->pixelsPerImage = pixelsPerImage;
	
	h_imageStack->redSubframes = redSubframes;
	h_imageStack->greenSubframes = greenSubframes;
	h_imageStack->blueSubframes = blueSubframes;

	h_imageStack->stackedRed = (float*) malloc(pixelsPerImage * sizeof(float));
	h_imageStack->stackedGreen = (float*) malloc(pixelsPerImage * sizeof(float));
	h_imageStack->stackedBlue = (float*) malloc(pixelsPerImage * sizeof(float));

	return h_imageStack;
}

/**
 * Initializes a copy of a Stack on the DEVICE from the HOST.
 * 
 * @param h_imageStack Initialized Stack on the HOST with populated attributes
 * @param h_lookupStack Initialized Stack on HOST with empty attributes
 * @return Pointer to a Stack on the DEVICE 
 */
Stack* initializeStackDEVICE(Stack* h_imageStack, Stack* h_lookupStack)
{
    // Allocate the Stack on the DEVICE
    Stack* d_imageStack;
    cudaMalloc(&d_imageStack, sizeof(Stack));

    // Populate the non-pointer Stack attributes
    cudaMemcpy(d_imageStack, h_imageStack, sizeof(Stack), cudaMemcpyHostToDevice);

    // Populate the pointer Stack attributes
    uint64_t N = h_imageStack->numberOfSubframes * h_imageStack->pixelsPerImage;

    uint16_t* d_redSubframes, * d_greenSubframes, * d_blueSubframes;
    cudaMalloc(&d_redSubframes, N * sizeof(uint16_t));
    cudaMalloc(&d_greenSubframes, N * sizeof(uint16_t));
    cudaMalloc(&d_blueSubframes, N * sizeof(uint16_t));

    cudaMemcpy(d_redSubframes, h_imageStack->redSubframes, 
		N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_greenSubframes, h_imageStack->greenSubframes, 
		N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blueSubframes, h_imageStack->blueSubframes, 
		N * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // Copy DEVICE pointers addresses to corresponding fields in d_imageStack
    cudaMemcpy(&(d_imageStack->redSubframes), &d_redSubframes, 
		sizeof(uint16_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_imageStack->greenSubframes), &d_greenSubframes, 
		sizeof(uint16_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_imageStack->blueSubframes), &d_blueSubframes, 
		sizeof(uint16_t*), cudaMemcpyHostToDevice);

	float *d_stackedRed, *d_stackedGreen, *d_stackedBlue;
	cudaMalloc(&d_stackedRed, N * sizeof(float));
	cudaMalloc(&d_stackedGreen, N * sizeof(float));
	cudaMalloc(&d_stackedBlue, N * sizeof(float));

	cudaMemcpy(&(d_imageStack->stackedRed), &d_stackedRed, 
		sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_imageStack->stackedGreen), &d_stackedGreen, 
		sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_imageStack->stackedBlue), &d_stackedBlue, 
		sizeof(float*), cudaMemcpyHostToDevice);

	/*
	d_imageStack resides on the DEVICE. If we try to copy data from the DEVICE 
	to the HOST by performing 
	cudaMemcpy(h_imageStack->array, d_imageStack->array, ...), we will 
	encounter a segmentation fault. d_imageStack-> deferences the pointer to 
	d_imageStack's block of memory so we can access its attributes. Since the 
	struct resides on VRAM, the CPU would be accessing illegal memory. To get 
	around this, we utilize a "lookup struct". This struct resides on the HOST 
	so that it can be deferenced but its pointer attributes contain VRAM 
	address values so that we can retrieve data using cudaMemcpy() 
	(e.g., cudaMemcpy(h_imageStack->array, h_lookupStack->array, ...))
	*/

	h_lookupStack->redSubframes = d_redSubframes;
	h_lookupStack->greenSubframes = d_greenSubframes;
	h_lookupStack->blueSubframes = d_blueSubframes;
	h_lookupStack->stackedRed = d_stackedRed;
	h_lookupStack->stackedGreen = d_stackedGreen;
	h_lookupStack->stackedBlue = d_stackedBlue;

    return d_imageStack;
}

/**
 * Sum stacking. Subframes in the stack are split up by color channel 
 * and stacked independently. Each pixel in the stack is summed. The increase 
 * in signal-to-noise ration (SNR) is proportional to 
 * sqrt(Number of subframes). The sum is performed and then normalized by the 
 * maximum pixel value out of all color channels and saved as a 32-bit floating 
 * point image. This function launches sumStackKernel @see stack_kernels.cu
 * 
 * @param h_imageStack Initialized Stack on the HOST
 * @return Stack with sum stacked subframes
 */
Stack* launchSumStack(Stack* h_imageStack)
{
	// Allocate the Stack on DEVICE
	Stack* h_lookupStack = (Stack*) malloc(sizeof(Stack));
	memcpy(h_lookupStack, h_imageStack, sizeof(Stack));
	Stack* d_imageStack = initializeStackDEVICE(h_imageStack, h_lookupStack);

	// Additional kernel parameters
	uint64_t* d_maximumPixel;
	cudaMalloc(&d_maximumPixel, sizeof(uint64_t));
	cudaMemset(d_maximumPixel, 0, sizeof(uint64_t));

	// DEVICE kernel launch configuration
	dim3 B(32, 32);
	dim3 G((h_imageStack->imageWidth + B.x - 1) / B.x, 
		(h_imageStack->imageHeight + B.y - 1) / B.y);
	sumStack <<< G, B >>> (d_imageStack, d_maximumPixel);
	cudaDeviceSynchronize();

	// Copy stacked frames back to HOST
	cudaMemcpy(h_imageStack->stackedRed, d_imageStack->stackedRed, 
		h_imageStack->pixelsPerImage * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_imageStack->stackedGreen, d_imageStack->stackedGreen, 
		h_imageStack->pixelsPerImage * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_imageStack->stackedBlue, d_imageStack->stackedBlue, 
		h_imageStack->pixelsPerImage * sizeof(float), cudaMemcpyDeviceToHost);

	// Free DEVICE allocation
	cudaFree(d_imageStack->redSubframes);
	d_imageStack->redSubframes = NULL;

	cudaFree(d_imageStack->greenSubframes);
	d_imageStack->greenSubframes = NULL;

	cudaFree(d_imageStack->blueSubframes);
	d_imageStack->blueSubframes = NULL;

	cudaFree(d_imageStack->stackedRed);
	d_imageStack->stackedRed = NULL;

	cudaFree(d_imageStack->stackedGreen);
	d_imageStack = NULL;

	cudaFree(d_imageStack->stackedBlue);
	d_imageStack->stackedBlue = NULL;

	cudaFree(d_imageStack);
	d_imageStack = NULL;
	
	return h_imageStack;
}

/**
 * Launches sigma clipping kernel.
 * 
 * @param h_imageStack Initialized Stack on the HOST
 * @param sigmaLow Sigma unit lower bound for rejection
 * @param sigmaHigh Sigma unit upper bound for rejection
 * @return Stack containing average stacked with sigma clipping rejection 
 * subframes
 */
Stack* launchSigmaClipping(Stack* h_imageStack, float sigmaLow, 
	float sigmaHigh)
{
	// Allocate the lookup struct on the HOST @see initializeStackDEVICE()
	Stack* h_lookupStack = (Stack*) malloc(sizeof(Stack));
	h_lookupStack->numberOfSubframes = h_imageStack->numberOfSubframes;
	h_lookupStack->imageWidth = h_imageStack->imageWidth;
	h_lookupStack->imageHeight = h_imageStack->imageHeight;
	h_lookupStack->pixelsPerImage = h_imageStack->pixelsPerImage;

	// Allocate the Stack on the DEVICE
	Stack* d_imageStack = initializeStackDEVICE(h_imageStack, h_lookupStack);

	// DEVICE kernel launch configuration
	dim3 B(32, 32);
	dim3 G((h_imageStack->imageWidth + B.x - 1) / B.x, 
		(h_imageStack->imageHeight + B.y - 1) / B.y);
	sigmaClipping <<< G, B >>> (d_imageStack, sigmaLow, sigmaHigh);
	cudaDeviceSynchronize();

	// Copy stacked frames back to HOST
	cudaMemcpy(h_imageStack->stackedRed, h_lookupStack->stackedRed, 
		h_imageStack->pixelsPerImage * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_imageStack->stackedGreen, h_lookupStack->stackedGreen, 
		h_imageStack->pixelsPerImage * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_imageStack->stackedBlue, h_lookupStack->stackedBlue, 
		h_imageStack->pixelsPerImage * sizeof(float), cudaMemcpyDeviceToHost);

	// Free arrays and null pointers
	cudaFree(h_lookupStack->redSubframes);
	cudaFree(h_lookupStack->greenSubframes);
	cudaFree(h_lookupStack->blueSubframes);
	cudaFree(h_lookupStack->stackedRed);
	cudaFree(h_lookupStack->stackedGreen);
	cudaFree(h_lookupStack->stackedBlue);
	cudaFree(d_imageStack);
	free(h_lookupStack);

	h_lookupStack->redSubframes = NULL;
	h_lookupStack->greenSubframes = NULL;
	h_lookupStack->blueSubframes = NULL;
	h_lookupStack->stackedRed = NULL;
	h_lookupStack->stackedGreen = NULL;
	h_lookupStack->stackedBlue = NULL;
	d_imageStack = NULL;
	h_lookupStack = NULL;

	return h_imageStack;	
}
