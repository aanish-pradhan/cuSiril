/**
 * Implements DEVICE kernels for image stacking.
 * 
 * @author Aanish Pradhan
 * @version 2023-11-20 
 */

// INCLUDE LIBRARIES
#include <math.h>
#include "lib/Stacking/stack.h"

// FUNCTIONS
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
__global__ void sumStackKernel(Stack* imageStack, uint64_t* maximumPixel)
{
	// 2D thread indexes
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	/*
	The subframes are flattened so we convert the 2D thread indexes into a 
	linear, 1D index.
	*/
	int pixelIndex = imageStack->imageWidth * y + x;

	// Legal thread index check
	if (x < imageStack->imageWidth && y < imageStack->imageHeight)
	{
		// Stacking
		for (uint64_t subframe = 0; subframe < imageStack->numberOfSubframes;
			subframe++)
		{
			/*
			The subframes are flattened so we offset pixelIndex to move to the 
			next subframe in the stack.
			*/
			uint64_t subframeOffset = imageStack->pixelsPerImage * subframe;

			imageStack->stackedRed[pixelIndex] += 
				imageStack->redSubframes[pixelIndex + subframeOffset];
			imageStack->stackedGreen[pixelIndex] += 
				imageStack->greenSubframes[pixelIndex + subframeOffset];
			imageStack->stackedBlue[pixelIndex] += 
				imageStack->blueSubframes[pixelIndex + subframeOffset];
		}
	}

	// Find normalization constant
	if (x == 0 & y == 0) // TODO: Optimize
	{
		for (uint64_t pixel = 0; pixel < imageStack->pixelsPerImage; pixel++)
		{
			
			if (imageStack->stackedRed[pixel] > *maximumPixel)
			{
				*maximumPixel = imageStack->stackedRed[pixel];
			}
			
			if (imageStack->stackedGreen[pixel] > *maximumPixel)
			{
				*maximumPixel = imageStack->stackedGreen[pixel];
			}
			
			if (imageStack->stackedBlue[pixel] > *maximumPixel)
			{
				*maximumPixel = imageStack->stackedBlue[pixel];
			}
		}
	}

	// Normalization
	imageStack->stackedRed[pixelIndex] /= *maximumPixel;
	imageStack->stackedGreen[pixelIndex] /= *maximumPixel;
	imageStack->stackedBlue[pixelIndex] /= *maximumPixel;
}
