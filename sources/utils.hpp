/**
 * @file    utils.hpp
 * 
 * @authors David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *
 * @date    2024-02-22
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <iostream>
#include <optional>
#include <string_view>

/**
 * @brief Structure for error information.
 */
struct ErrorInfo
{
  float       maxError{};    ///< Maximum error value.
  std::size_t maxErrorIdx{}; ///< Index of the maximum error value.
};

/**
 * @brief Print 2D array to the standard output.
 * @param array Pointer to the array.
 * @param edgeSize Size of the edge of the 2D array.
 */
void printArray2d(const float* array, std::size_t edgeSize);

/**
 * @brief Save 2D array as an image.
 * @param fileName Name of the file.
 * @param array Pointer to the array.
 * @param edgeSize Size of the edge of the 2D array.
 * @param normRange Optional range of the normalization values in the array.
 */
void saveAsImage(std::string_view                       fileName,
                 const float*                           array,
                 std::size_t                            edgeSize,
                 std::optional<std::pair<float, float>> normRange = std::nullopt);

/**
 * @brief Verify results of the parallel computation.
 * @param result Pointer to the parallel result.
 * @param refResult Pointer to the reference result.
 * @param gridPointCount Number of grid points.
 * @param outAbsDiff Pointer to the output array for absolute differences.
 * @param epsilon Maximum allowed deviation.
 * @return Pair of boolean value and error information.
 */
std::pair<bool, ErrorInfo> verifyResults(const float* result,
                                         const float* refResult,
                                         std::size_t  gridPointCount,
                                         float*       outAbsDiff,
                                         float        epsilon = 0.001f);

#endif /* UTILS_HPP */
