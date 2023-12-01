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
} 
#include "stack_kernels.cu"

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
 * @return Pointer to a Stack on HOST
 */
extern "C" Stack* initializeStack(uint64_t numberOfSubframes, uint64_t imageWidth, 
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
 * Sum stacking. Subframes in the stack are split up by color channel 
 * and stacked independently. Each pixel in the stack is summed. The increase 
 * in signal-to-noise ration (SNR) is proportional to 
 * sqrt(Number of subframes). The sum is performed and then normalized by the 
 * maximum pixel value out of all color channels and saved as a 32-bit floating 
 * point image. This function launches sumStackKernel @see stack_kernels.cu
 * 
 * @param h_imageStack Initialized Stack on HOST
 * @return Stack with sum stacked subframes
 */
Stack* sumStack(Stack* h_imageStack)
{
	uint64_t N = h_imageStack->numberOfSubframes * h_imageStack->pixelsPerImage;

	// Allocate the Stack on DEVICE
	Stack* d_imageStack;
	cudaMalloc(&d_imageStack, sizeof(Stack));

	// Populate the Stack attributes
	cudaMemset(&d_imageStack->numberOfSubframes, 
		h_imageStack->numberOfSubframes, sizeof(uint64_t));
	cudaMemset(&d_imageStack->imageWidth, h_imageStack->imageWidth, 
		sizeof(uint64_t));
	cudaMemset(&d_imageStack->imageHeight, h_imageStack->imageHeight, 
		sizeof(uint64_t));
	cudaMemset(&d_imageStack->pixelsPerImage, h_imageStack->pixelsPerImage, 
		sizeof(uint64_t));

	cudaMalloc(&d_imageStack->redSubframes, N * sizeof(uint16_t));
	cudaMalloc(&d_imageStack->greenSubframes, N * sizeof(uint16_t));
	cudaMalloc(&d_imageStack->blueSubframes, N * sizeof(uint16_t));

	cudaMemcpy(d_imageStack->redSubframes, h_imageStack->redSubframes, 
		N * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_imageStack->greenSubframes, h_imageStack->greenSubframes, 
		N * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_imageStack->blueSubframes, h_imageStack->blueSubframes, 
		N * sizeof(uint16_t), cudaMemcpyHostToDevice);

	cudaMalloc(&d_imageStack->stackedRed, N * sizeof(float));
	cudaMalloc(&d_imageStack->stackedGreen, N * sizeof(float));
	cudaMalloc(&d_imageStack->stackedBlue, N * sizeof(float));

	// Additional kernel parameters
	uint64_t* d_maximumPixel;
	cudaMalloc(&d_maximumPixel, sizeof(uint64_t));
	cudaMemset(d_maximumPixel, 0, sizeof(uint64_t));

	// DEVICE kernel launch configuration
	dim3 B(32, 32);
	dim3 G((h_imageStack->imageWidth + B.x - 1) / B.x, 
		(h_imageStack->imageHeight + B.y - 1) / B.y);
	sumStackKernel <<< G, B >>> (d_imageStack, d_maximumPixel);
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
