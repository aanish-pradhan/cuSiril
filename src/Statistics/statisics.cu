/**
 * Implements DEVICE kernels for computing basic statistics.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-01
 */

// INCLUDE LIBRARIES
#include <math.h>
#include "quickselect.cuh"
#include "statistics.cuh"

// KERNELS
/**
 * Computes the median of a dataset.
 * 
 * @param data Data to compute the median of
 * @param n Number of datapoints
 */
__device__ float median(float data[], uint64_t n)
{
	uint64_t medianIndex = (n + 1) / 2;
	float median = quickSelect(data, 0, n - 1, medianIndex);
	return median;
}

/**
 * Computes the standard deviation of a dataset.
 * 
 * @param data Data to compute the standard deviation of
 * @param n Number of datapoints
 * @param center Measure of central tendency (i.e. mean, median, mode) of the 
 * dataset
 * @param type Population or sample (default) standard deviation
 */
__device__ float stdev(float data[], uint64_t n, float center, char* type)
{
	float sum = 0.0;
	for (int i = 0; i < n; i++)
	{
		sum += powf32((data[i] - center), 2);
	}
	
	(strcmp(type, "sample") == 0) ? sum /= (n - 1) : sum /= n;

	return sqrtf32(sum);
}
