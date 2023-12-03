#ifndef CALIBRATE_KERNELS_H
#define CALIBRATE_KERNELS_H
/**
 * Defines DEVICE kernels for calibration frame subtraction.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-02 
 */

// INCLUDE LIBRARIES
#include "stack.h"

// KERNELS
/**
 * Normalizes a stack of light frames by UINT16_MAX.
 * 
 * @param lightFrames Pointer to a Stack with unnormalized, uint16 light frames
 */
__global__ void normalizeLightFrames(Stack* lightFrames);

/**
 * Subtracts a master calibration frame from other calibration frames or light
 * frames.
 * 
 * @param parentFrame The calibration frames or light frames to be subtracted
 * from
 * @param calibrationFrame The calibration frame to subtract
 */
__global__ void subtractMasterCalibrationFrame(Stack* parentFrame, 
	Stack* calibrationFrame);

#endif