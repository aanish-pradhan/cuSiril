/**
 * Implements functions for calibration frame subtraction. 
 * 
 * @author Aanish Pradhan
 * @version 2023-11-28
 */

// INCLUDE LIBRARIES
#include "calibrate.h"
#include "calibrate_kernels.cu"
#include "stack.cuh"
#include <math.h>

// DEFINITIONS

// FUNCTIONS
Stack* calibrate(bool useBiasFrames, Stack* biasFrames, bool useDarkFrames, 
	Stack* darkFrames, bool useFlatFrames, Stack* flatFrames, 
		Stack* lightFrames)
{
	if (useBiasFrames || useDarkFrames || useFlatFrames)
	{	
		normalizeLightFrames(lightFrames);


		// Check if Bias frames are provided
		if (useBiasFrames == true)
		{
			sigmaClipping(biasFrames, 3.0, 3.0);	
			subtractMasterCalibrationFrame(lightFrames, biasFrames);
		}

		// Check if Dark frames are provided
		if (useDarkFrames == true)
		{
			if (useBiasFrames == true)
			{
				subtractMasterCalibrationFrame(darkFrames, biasFrames);
			}
			sigmaClipping(darkFrames, 3.0, 3.0);
			subtractMasterCalibrationFrame(lightFrames, darkFrames);
		}

		// Check if Flat frames are provided
		if (useFlatFrames == true)
		{
			if (useBiasFrames == true)
			{
				subtractMasterCalibrationFrame(flatFrames, biasFrames);
			}
			sigmaClipping(flatFrames, 3.0, 3.0);
			divideFlatFrames(lightFrames, flatFrames);
		}
	}

	return lightFrames;
}

