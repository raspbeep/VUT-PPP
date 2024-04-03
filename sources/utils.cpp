/**
 * @file    utils.cpp
 * 
 * @authors David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *
 * @date    2024-02-22
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#ifdef _MSC_VER
# define STBI_MSC_SECURE_CRT
#endif
#include <stb_image_write.h>
#ifdef _MSC_VER
# undef STBI_MSC_SECURE_CRT
#endif
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "utils.hpp"

void printArray2d(const float* array, std::size_t edgeSize)
{
  for(std::size_t i = 0; i < edgeSize; ++i)
  {
    fmt::print("[Row {}]: ", i);
    
    for(std::size_t j = 0; j < edgeSize; ++j)
    {
      fmt::print("{}{}", array[i * edgeSize + j], (j + 1 < edgeSize) ? " " : "\n");
    }
  }
}

void saveAsImage(std::string_view                       fileName,
                 const float*                           array,
                 std::size_t                            edgeSize,
                 std::optional<std::pair<float, float>> normRange)
{
  static constexpr unsigned paletteWidth{18};

  // Allocate buffer for pixel data in the image
  std::vector<std::uint8_t> imageData(3 * (edgeSize + paletteWidth) * edgeSize, std::uint8_t{});

  // Find minimum and maximum values in the data for normalization
  float minValue{};
  float maxValue{};

  if(!normRange.has_value())
  {
    auto [minPtr, maxPtr] = std::minmax_element(array, array + edgeSize * edgeSize);

    minValue = *minPtr;
    maxValue = *maxPtr;
  }
  else
  {
    minValue = normRange->first;
    maxValue = normRange->second;
  }

  // Normalize the values and compute their color representation according to
  // the MATLAB HSV palette
  for(unsigned i = 0; i < edgeSize; ++i)
  {
    // Write values to image as usual and
    for(unsigned j = 0; j < edgeSize; ++j)
    {
      std::size_t srcIdx = i * edgeSize + j;
      std::size_t dstIdx = i * (edgeSize + paletteWidth) + j;

      float normalValue = (array[srcIdx] - minValue) / (maxValue - minValue);

      float red   = (normalValue < 0.5f) ? (-6.0f * normalValue + 67.0f / 32.0f) : ( 6.0f * normalValue -  79.0f / 16.0f);
      float green = (normalValue < 0.4f) ? ( 6.0f * normalValue -  3.0f / 32.0f) : (-6.0f * normalValue +  79.0f / 16.0f);
      float blue  = (normalValue < 0.7f) ? ( 6.0f * normalValue - 67.0f / 32.0f) : ( 6.0f * normalValue + 195.0f / 32.0f);

      imageData[3 * dstIdx + 0] = static_cast<std::uint8_t>(std::clamp(red,   0.f, 1.f) * 255.0f);
      imageData[3 * dstIdx + 1] = static_cast<std::uint8_t>(std::clamp(green, 0.f, 1.f) * 255.0f);
      imageData[3 * dstIdx + 2] = static_cast<std::uint8_t>(std::clamp(blue,  0.f, 1.f) * 255.0f);
    }

    // add palette strip at the end (with 2px padding on left side).
    for(unsigned j = 2; j < paletteWidth; ++j)
    {
      std::size_t dstIdx = i * (edgeSize + paletteWidth) + j + edgeSize;

      float normalValue = 1.0f - static_cast<float>(i) * (1.0f / static_cast<float>(edgeSize));

      float red   = (normalValue < 0.5f) ? (-6.0f * normalValue + 67.0f / 32.0f) : ( 6.0f * normalValue -  79.0f / 16.0f);
      float green = (normalValue < 0.4f) ? ( 6.0f * normalValue -  3.0f / 32.0f) : (-6.0f * normalValue +  79.0f / 16.0f);
      float blue  = (normalValue < 0.7f) ? ( 6.0f * normalValue - 67.0f / 32.0f) : ( 6.0f * normalValue + 195.0f / 32.0f);

      imageData[3 * dstIdx + 0] = static_cast<std::uint8_t>(std::clamp(red,   0.0f, 1.0f) * 255.0f);
      imageData[3 * dstIdx + 1] = static_cast<std::uint8_t>(std::clamp(green, 0.0f, 1.0f) * 255.0f);
      imageData[3 * dstIdx + 2] = static_cast<std::uint8_t>(std::clamp(blue,  0.0f, 1.0f) * 255.0f);
    }
  }

  // Write RGB pixel data into the *.png file.
  stbi_write_png(fileName.data(), static_cast<int>(edgeSize + paletteWidth), static_cast<int>(edgeSize), 3,
                 imageData.data(), static_cast<int>(3 * (edgeSize + paletteWidth)));
}

std::pair<bool, ErrorInfo> verifyResults(const float* result,
                                         const float* refResult,
                                         std::size_t  gridPointCount,
                                         float*       outAbsDiff,
                                         float        epsilon)
{
  ErrorInfo errorInfo{};

  for(std::size_t i = 0; i < gridPointCount; ++i)
  {
    outAbsDiff[i] = std::abs(result[i] - refResult[i]);

    if(outAbsDiff[i] > errorInfo.maxError)
    {
      errorInfo.maxError    = outAbsDiff[i];
      errorInfo.maxErrorIdx = i;
    }
  }

  return std::make_pair(errorInfo.maxError <= epsilon, errorInfo);
}