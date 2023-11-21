#ifndef CALIBRATE_H
#define CALIBRATE_H
/**
 * Defines functions for calibration frame subtration.
 * 
 * @author Aanish Pradhan
 * @version 2023-11-20
 */

// INCLUDE LIBRARIES
#include "../Stacking/stack.h"
#include <stdbool.h>

// FUNCTIONS
Stack* subtractBDF(bool haveBiasFrames, Stack* biasFrames, bool haveDarkFrames, 
	Stack* darkFrames, bool haveFlatFrames, Stack* flatFrames, 
		Stack* lightFrames);

#endif