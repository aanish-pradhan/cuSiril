#ifndef STATISTICS_H
#define STATISTICS_H
/**
 * Defines DEVICE kernels for computing basic statistics.
 * 
 * @author Aanish Pradhan
 * @version 2023-12-02 
 */

// INCLUDE LIBRARIES
#include <stdint.h>

// KERNELS
/**
 * Computes the median of a dataset.
 * 
 * @param data Data to compute the median of
 * @param n Number of datapoints
 */
__device__ float median(float data[], uint64_t n);

/**
 * Computes the standard deviation of a dataset.
 * 
 * @param data Data to compute standard deviation of
 * @param n Number of datapoints
 * @param center Measure of central tendency (i.e. mean, median, mode) of the 
 * dataset
 * @param type Population or sample (default) standard deviation
 */
__device__ float stdev(float data[], uint64_t n, float center, char* type);

#endif
