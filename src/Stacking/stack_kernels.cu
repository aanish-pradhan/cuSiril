/**
 * Implements DEVICE kernels for image stacking.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-02 
 */

// INCLUDE LIBRARIES
#include <math.h>
extern "C"
{
	#include "stack.cuh"
	#include "statistics.cuh"	
}

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
__global__ void sumStack(Stack* imageStack, uint64_t* maximumPixel)
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

/**
 * Decides if a pixel should be rejected or not for Sigma Clipping.
 * 
 * @param pxiel Pixel value to evalulate
 * @param center 
 * @param standardDeviation
 * @return True if the pixel should be kept, false if the pixel should be 
 * rejected
 */
__device__ bool keepPixel(float pixel, float center, float standardDeviation,
	float sigmaLow, float sigmaHigh)
{
	float sigmaUnitLow = center - standardDeviation * sigmaLow;
	float sigmaUnitHigh = center + standardDeviation * sigmaHigh;

	if (pixel < sigmaUnitLow || pixel > sigmaUnitHigh)
	{
		return false;
	}
	else
	{
		return true;
	}
}

/**
 * Sigma Clipping kernel. This is an iterative algorithm which will reject 
 * pixels whose distance from median will be farthest than two given values in 
 * sigma units (sigmaLow, sigmaHigh). The improvement in SNR is proportional to 
 * sqrt(Number of subframes).
 * 
 * @param imageStack The image stack @see stack.h
 * @param sigmaLow The lower standard deviation bound
 * @param sigmaHigh The higher standard deviation bound
 */
__global__ void sigmaClipping(Stack* imageStack, float sigmaLow, 
	float sigmaHigh)
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
		uint64_t numberOfSubframes = imageStack->numberOfSubframes;

		float *redPixelArray = (float*) malloc(numberOfSubframes * sizeof(float));
		float *greenPixelArray = (float*) malloc(numberOfSubframes * sizeof(float));
		float *bluePixelArray = (float*) malloc(numberOfSubframes * sizeof(float));
		
		for (uint64_t subframe = 0; subframe < numberOfSubframes; subframe++)
		{
			uint64_t subframeOffset = imageStack->pixelsPerImage * subframe;

			redPixelArray[subframe] = imageStack->redSubframes[pixelIndex + subframeOffset];
			greenPixelArray[subframe] = imageStack->greenSubframes[pixelIndex + subframeOffset];
			bluePixelArray[subframe] = imageStack->blueSubframes[pixelIndex + subframeOffset];
		}

		// Compute median
		float redPixelMedian = median(redPixelArray, numberOfSubframes);
		float greenPixelMedian = median(greenPixelArray, numberOfSubframes);
		float bluePixelMedian = median(bluePixelArray, numberOfSubframes);

		// Compute standard deviation
		char stdevType[] = "sample";
		float redPixelStdev = stdev(redPixelArray, numberOfSubframes, 
			redPixelMedian, stdevType);
		float greenPixelStdev = stdev(greenPixelArray, numberOfSubframes, 
			greenPixelMedian, stdevType);
		float bluePixelStdev = stdev(bluePixelArray, numberOfSubframes, 
			bluePixelMedian, stdevType);
		
		// Stacking
		for (uint64_t subframe = 0; subframe < numberOfSubframes;
			subframe++)
		{
			uint64_t subframeOffset = imageStack->pixelsPerImage * subframe;

			float redPixel = imageStack->redSubframes[pixelIndex + subframeOffset];
			float greenPixel = imageStack->greenSubframes[pixelIndex + subframeOffset];
			float bluePixel = imageStack->blueSubframes[pixelIndex + subframeOffset];

			if (keepPixel(redPixel, redPixelMedian, redPixelStdev, sigmaLow, sigmaHigh))
			{
				imageStack->stackedRed[pixelIndex] += redPixel;
			}

			if (keepPixel(greenPixel, greenPixelMedian, greenPixelStdev, sigmaLow, sigmaHigh))
			{
				imageStack->stackedGreen[pixelIndex] += greenPixel;
			}

			if (keepPixel(bluePixel, bluePixelMedian, bluePixelStdev, sigmaLow, sigmaHigh))
			{
				imageStack->stackedBlue[pixelIndex] += bluePixel;
			}
		}

		// Free arrays
		free(redPixelArray);
		free(greenPixelArray);
		free(bluePixelArray);
	}
}
