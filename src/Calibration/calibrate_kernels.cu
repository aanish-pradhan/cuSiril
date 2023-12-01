/**
 * Implements DEVICE kernels for calibration frame subtraction.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-01
 */

// INCLUDE LIBRARIES
#include "calibrate.cuh"
#include <math.h>

// KERNELS
/**
 * Normalizes a stack of light frames by UINT16_MAX.
 * 
 * @param lightFrames Pointer to a Stack with unnormalized, uint16 light frames
 */
__global__ void normalizeLightFrames(Stack* lightFrames)
{
	// 2D thread indexes
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	/*
	The subframes are flattened so we convert the 2D thread indexes into a 
	linear, 1D index.
	*/
	int pixelIndex = lightFrames->imageWidth * y + x;

	if (x < lightFrames->imageWidth && y < lightFrames->imageHeight)
	{
		for (uint64_t subframe = 0; subframe < lightFrames->numberOfSubframes; 
			subframe++)
		{
			/*
			The subframes are flattened so we offset pixelIndex to move to the 
			next subframe in the stack.
			*/
			uint64_t subframeOffset = lightFrames->pixelsPerImage * subframe;

			lightFrames->redSubframes[pixelIndex] /= UINT16_MAX;
			lightFrames->greenSubframes[pixelIndex] /= UINT16_MAX;
			lightFrames->blueSubframes[pixelIndex] /= UINT16_MAX;
		}
	}
}

/**
 * Subtracts a master calibration frame from other calibration frames or light
 * frames.
 * 
 * @param parentFrame The calibration frames or light frames to be subtracted
 * from
 * @param calibrationFrame The calibration frame to subtract
 */
__global__ void subtractMasterCalibrationFrame(Stack* parentFrame, 
	Stack* calibrationFrame)
{
	// TODO
}
