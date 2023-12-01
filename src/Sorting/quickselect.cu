/**
 * Implements kernels for Quickselect algorithm.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-01
 */

// INCLUDE LIBRARIEs
extern "C"
{
	#include "quickselect.cuh"
}

// KERNELS
/**
 * Swaps two numbers.
 * 
 * @param a First number to be swapped
 * @param b Second number to be swapped 
 */
__device__ void swap(float* a, float* b)
{
	float temp = *a;
	*a = *b;
	*b = temp;
}

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
__device__ int partition(float array[], int startingIndex, int endingIndex)
{
	float pivot = array[endingIndex];
	int i = startingIndex - 1;

	for (int j = startingIndex; j < endingIndex; j++)
	{
		if (array[j] < pivot)
		{
			i++;
			swap(&array[i], &array[j]);
		}
	}

	swap(&array[i + 1], &array[endingIndex]);
	return i + 1;
}

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
	int k)
{
	if (startingIndex <= endingIndex)
	{
		int pivotIndex = partition(array, startingIndex, endingIndex);

		if (pivotIndex == k - 1)
		{
			return array[pivotIndex];
		}
		else if (pivotIndex < k - 1)
		{
			return quickSelect(array, pivotIndex + 1, endingIndex, k);
		}
		else
		{
			return quickSelect(array, startingIndex, pivotIndex - 1, k);
		}
	}

	return -1.0f;
}
