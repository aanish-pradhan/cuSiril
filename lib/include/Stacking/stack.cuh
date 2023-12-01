#ifndef STACK_KERNELS_H
#define STACK_KERNELS_H
/**
 * Defines DEVICE kernels for image stacking.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-01 
 */

// INCLUDE LIBRARIES
#include "stack.h"
#include <stdbool.h>

// KERNELS
/**
 * Sum stacking kernel. Subframes in the stack are split up by color channel 
 * and stacked independently. Each pixel in the stack is summed. The increase 
 * in signal-to-noise ration (SNR) is proportional to 
 * sqrt(Number of subframes). The sum is performed and then normalized by the 
 * maximum pixel value out of all color channels and saved as a 32-bit floating 
 * point image. This kernel is launched by sumStack() @see stack.cu
 * 
 * @param imageStack The image stack @see stack.h
 * @param maximumPixel The normalization constant
 */
__global__ void sumStackKernel(Stack* imageStack, uint64_t* maximumPixel);


/**
 * Decides if a pixel should be rejected or not for Sigma Clipping.
 * 
 * @param value Pixel value to evalulate
 * @param center 
 * @param standardDeviation
 * @return True if the pixel should be kept, false if the pixel should be 
 * rejected
 */
__device__ bool keepPixel(float value, float center, float standardDeviation, 
	float sigmaLow, float sigmaHigh);


/**
 * Sigma Clipping kernel. This is an iterative algorithm which will reject 
 * pixels whose distance from median will be farthest than two given values in 
 * sigma units (sigmaLow, sigmaHigh).
 * 
 * @param imageStack The image stack @see stack.h
 * @param sigmaLow The lower standard deviation bound
 * @param sigmaHigh The higher standard deviation bound
 */
__global__ void sigmaClippingKernel(Stack* imageStack, float sigmaLow, 
	float sigmaHigh);

#endif