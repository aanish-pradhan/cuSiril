// INCLUDE LIBRARIES
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Sum stacking kernel. Subframes in the stack are split up by color channel
 * and each color channel is stacked independently. Each pixel in the stack is 
 * summed. The increase in signal-to-noise ratio (SNR) is proportional to the 
 * square root of the number of subframes. The sum is performed in a 64-bit 
 * unsigned integer, normalized by the maximum pixel value out of all color 
 * channels and saved as a 32-bit floating point image.
 * 
 * @param numberOfSubframes The number of subframes in the stack
 * @param imageWidth Width of the image (pixels)
 * @param imageHeight Height of the image (pixels)
 * @param d_redChannelX DEVICE input array with consecutive flattened red 
 * channels from each subframe
 * @param d_greenChannelX DEVICE input array with consecutive flattened green 
 * channels from each subframe
 * @param d_blueChannelX DEVICE input array with consecutive flattened blue
 * channels from each subframe
 * @param d_redChannelY DEVICE output array with stacked red channels
 * @param d_greenChannelY DEVICE output array with stakced green channels
 * @param d_blueChannelY DEVICE output array with stacked blue channels
 */
__global__ void sumStack_kernel(uint64_t numberOfSubframes, uint64_t imageWidth, 
	uint64_t imageHeight, uint16_t* d_redChannelX, uint16_t* d_greenChannelX, 
		uint16_t* d_blueChannelX, float* d_redChannelY, float* d_greenChannelY, 
			float* d_blueChannelY, uint16_t* maximumPixel)
{
	// 2D thread indexes
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	/*
	Since the images are flattened, we convert the 2D thread indexes into a 
	single 1D, linear thread index.
	*/
	int pixelIndex = imageWidth * y + x;

	if (x < imageWidth && y < imageHeight) // Legal thread index check
	{
		// Pixel sums
		uint64_t redSum = 0;
		uint64_t greenSum = 0;
		uint64_t blueSum = 0;
		
		// Maximum pixel values
		uint16_t redPixelMax = 0;
		uint16_t greenPixelMax = 0;
		uint16_t bluePixelMax = 0;

		// Stacking
		for (uint64_t subframe = 0; subframe < numberOfSubframes; subframe++)
		{
			/*
			Since the images are flattened, we offset pixelIndex to move to the
			next subframe in the stack.
			*/
			uint64_t subframeOffset = imageWidth * imageHeight * subframe;

			// Fetch current pixel value
			uint16_t redPixel = d_redChannelX[pixelIndex + subframeOffset];
			uint16_t greenPixel = d_greenChannelX[pixelIndex + subframeOffset];
			uint16_t bluePixel = d_blueChannelX[pixelIndex + subframeOffset];

			// Update maximum pixel value
			(redPixel > redPixelMax) ? redPixelMax = redPixel : redPixel;
			(greenPixel > greenPixelMax) ? greenPixelMax = greenPixel : 
				greenPixel;
			(bluePixel > bluePixelMax) ? bluePixelMax = bluePixel : bluePixel;

			// Sum pixel values
			redSum += redPixel;
			greenSum += greenPixel;
			blueSum += bluePixel;
		}

		// TODO: Maximum pixel value
	}
}

void sumStack(uint64_t numberOfSubframes, uint64_t imageWidth, 
	uint64_t imageHeight, uint16_t* h_redChannelX, uint16_t* h_greenChannelX, 
		uint16_t* h_blueChannelX, float* h_redChannelY, float* h_greenChannelY, 
			float* h_blueChannelY)
{
	int pixelsPerImage = imageWidth * imageHeight;
	int N = pixelsPerImage * numberOfSubframes;

	// Copy subframes
	uint16_t *d_redChannelX, *d_greenChannelX, *d_blueChannelX;
	cudaMalloc(&d_redChannelX, N * sizeof(uint16_t));
	cudaMalloc(&d_greenChannelX, N * sizeof(uint16_t));	
	cudaMalloc(&d_blueChannelX, N * sizeof(uint16_t));

	cudaMemcpy(d_redChannelX, h_redChannelX, N * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_greenChannelX, h_greenChannelX, N * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blueChannelX, h_blueChannelX, N * sizeof(uint16_t), cudaMemcpyHostToDevice);

	// Allocate arrays for stacked frame
	float *d_redChannelY, *d_greenChannelY, *d_blueChannelY;
	cudaMalloc(&d_redChannelY, pixelsPerImage * sizeof(float));
	cudaMalloc(&d_greenChannelY, pixelsPerImage * sizeof(float));
	cudaMalloc(&d_blueChannelY, pixelsPerImage * sizeof(float));

	// Kernel launch config
	dim3 B(16, 16);
	dim3 G((imageWidth + B.x - 1) / B.x, (imageHeight + B.y - 1) / B.y);
}


