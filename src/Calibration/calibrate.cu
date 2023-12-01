/**
 * Implements functions for calibration frame subtraction. 
 * 
 * @author Aanish Pradhan
 * @version 2023-11-28
 */

// INCLUDE LIBRARIES
#include "calibrate.h"
#include "calibrate_kernels.cu"
#include <math.h>

// DEFINITIONS

// FUNCTIONS
Stack* calibrate(bool useBiasFrames, Stack* biasFrames, bool useDarkFrames, 
	Stack* darkFrames, bool useFlatFrames, Stack* flatFrames, 
		Stack* lightFrames)
{
	if (useBiasFrames || useDarkFrames || useFlatFrames)
	{	
		
		
		
		
		
		
		// Normalize Light frames



		// Check if Bias frames are provided
		if (useBiasFrames == true)
		{
			// TODO: Stack Bias frames with specified method
			// TODO: Subtract master Bias frame from Light frames
		}

		// Check if Dark frames are provided
		if (useDarkFrames == true)
		{
			if (useBiasFrames == true)
			{
				// TODO: Subtract master Bias frame from Dark frames
			}
			// TODO: Stack (corrected) Dark frames with specified method
			// TODO: Subtract master Dark frame from Light frames 
		}

		// Check if Flat frames are provided
		if (haveFlatFrames == true)
		{
			if (haveBiasFrames == true)
			{
				// TODO: Subtract master Bias frame from Flat frames
			}
			// TODO: Stack (corrected) Flat frames with specified method
			// TODO: Divide Light frames by master Flat frame
		}
	}

	return lightFrames;
}

