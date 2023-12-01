#ifndef QUICKSELECT_H
#define QUICKSELECT_H
/**
 * Defines DEVICE kernels for Quickselect algorithm.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-01
 */

// KERNELS
/**
 * Swaps two numbers.
 * 
 * @param a First number to be swapped
 * @param b Second number to be swapped 
 */
__device__ void swap(float* a, float* b);

/**
 * Parittions an array based on a given pivot element. Rearranges elements so 
 * that elements smaller than the pivot are placed to the left and elements 
 * greater that the pivot are placed to the right.
 * 
 * @param array The array to be partitioned
 * @param startingIndex The starting index of the partition range
 * @param endingIndex The ending index of the partition range
 * @return The index of the pivot element after partitioning
 */
__device__ int partition(float array[], int startingIndex, int endingIndex);

/**
 * Find the k-th smallest element in a given, unsorted array using the 
 * Quickselect algorithm.
 * 
 * @param array The array to search for the k-th smallest element
 * @param startingIndex The starting index of the search range
 * @param endingIndex The ending index of the search range
 * @param k The k-th smallest element to find (i.e. first (1) smallest, second 
 * (2) smallest, etc.)
 * @return The k-th smallest element in the array
 */
__device__ float quickSelect(float array[], int startingIndex, int endingIndex, 
	int k);

#endif
