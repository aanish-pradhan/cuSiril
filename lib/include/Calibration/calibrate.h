#ifndef CALIBRATE_H
#define CALIBRATE_H
/**
 * Defines functions for calibration frame subtration.
 * 
 * @author Aanish Pradhan
 * @version 2023-11-28
 */

// INCLUDE LIBRARIES
#include "../Stacking/stack.h"
#include <stdbool.h>

// FUNCTIONS
Stack* calibrate(bool useBiasFrames, Stack* biasFrames, bool useDarkFrames, 
	Stack* darkFrames, bool useFlatFrames, Stack* flatFrames, 
		Stack* lightFrames);

#endif