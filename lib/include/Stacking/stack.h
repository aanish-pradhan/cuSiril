#ifndef STACK_H
#define STACK_H
/**
 * Defines objects and functions for image stacking.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-02
 */

// INCLUDE LIBRARIES
#include <stdint.h>

// STRUCT DEFINITIONS
/**
 * struct for an image stack. A "stack" is a set of subframes to be stacked 
 * (i.e. combined) into a single image. The RGB channels of a stacked image are 
 * stacked independently of each other. 
 */
typedef struct Stack_s {
	uint64_t numberOfSubframes; // Number of subframes in image stack
	uint64_t imageWidth; // Subframe image width (px)
	uint64_t imageHeight; // Subframe image height (px)
	uint64_t pixelsPerImage; // Pixels per image in each frame
	uint16_t* redSubframes; // Red channels of flattened subframes
	uint16_t* greenSubframes; // Green channels of flattened subframes
	uint16_t* blueSubframes; // Blue channels of flattened subframes
	float* stackedRed; // Stacked image's red channel
	float* stackedGreen; // Stacked image's green channel
	float* stackedBlue; // Stacked image's blue channel
} Stack;

// FUNCTIONS
/**
 * Initializes a Stack with unstacked subframes.
 * 
 * @param numberOfSubframes Number of subframes in image stack
 * @param imageWidth Subframe image width (px)
 * @param imageHeight Subframe image height (px)
 * @param redSubframes Red channels of flattened subframes
 * @param greenSubframes Green channels of flattened subframes
 * @param blueSubframes Blue channels of flattened subframes
 * @return Pointer to a Stack on the HOST
 */
Stack* initializeStackHOST(uint64_t numberOfSubframes, uint64_t imageWidth, 
	uint64_t imageHeight, uint16_t* redSubframes, uint16_t* greenSubframes, 
		uint16_t* blueSubframes);

/**
 * Initializes a copy of a Stack on the DEVICE from the HOST.
 * 
 * @param h_imageStack Initialized Stack on the HOST with populated attributes
 * @param h_lookupStack Initialized Stack on HOST with empty attributes
 * @return Pointer to a Stack on the DEVICE 
 */
Stack* initializeStackDEVICE(Stack* h_imageStack, Stack* h_lookupStack);

/**
 * Sum stacking. Subframes in the stack are split up by color channel 
 * and stacked independently. Each pixel in the stack is summed. The increase 
 * in signal-to-noise ration (SNR) is proportional to 
 * sqrt(Number of subframes). The sum is performed and then normalized by the 
 * maximum pixel value out of all color channels and saved as a 32-bit floating 
 * point image. This function copies data to the DEVICE and launches 
 * sumStackKernel @see stack_kernels.cu
 * 
 * @param h_imageStack Initialized Stack on the HOST
 * @return Stack with sum stacked subframes
 */
Stack* launchSumStack(Stack* h_imageStack);

/**
 * Launches sigma clipping kernel.
 * 
 * @param h_imageStack Initialized Stack on the HOST
 * @return Stack containing average stacked with sigma clipping rejection 
 * subframes
 */
Stack* launchSigmaClipping(Stack* h_imageStack, float sigmaLow, 
	float sigmaHigh);

#endif