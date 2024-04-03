/**
 * @file    HeatSolverBase.hpp
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2022-02-03
 */

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>

#include <fmt/format.h>

#include "Hdf5Handle.hpp"
#include "HeatSolverBase.hpp"

HeatSolverBase::HeatSolverBase(const SimulationProperties& simulationProps,
                               const MaterialProperties&   materialProps)
: mSimulationProps(simulationProps),
  mMaterialProps(materialProps)
{}

bool HeatSolverBase::shouldPrintProgress(std::size_t iteration) const
{
  const std::size_t nTotalIters = mSimulationProps.getNumIterations();

  return !((iteration % (nTotalIters > 10 ? (nTotalIters / 10) : 1) != 0 && iteration + 1 != nTotalIters) ||
             mSimulationProps.isBatchMode());
}

bool HeatSolverBase::shouldStoreData(std::size_t iteration) const
{
  return (iteration % mSimulationProps.getWriteIntensity() == 0);
}

void HeatSolverBase::printProgressReport(std::size_t iteration, float middleColAvgTemp)
{
  if(!shouldPrintProgress(iteration))
  {
    return;
  }

  double progress = static_cast<double>((iteration + 1) * 100)
                      / static_cast<double>(mSimulationProps.getNumIterations());

  fmt::print("Progress {:3}% (Average Temperature {:.2f} degrees)\n", progress, middleColAvgTemp);
}

void HeatSolverBase::printFinalReport(double totalTime, float middleColAvgTemp) const
{
  const std::string_view codeType = getCodeType();

  if (!mSimulationProps.isBatchMode())
  {
    fmt::print("====================================================\n"
               "Execution time of \"{}\" version: {}s\n"
               "====================================================\n\n",
               codeType, totalTime);
  }
  else
  {
    fmt::print("{};{};{};{};{}\n",
               mSimulationProps.getOutputFileName(codeType),
               codeType,
               middleColAvgTemp,
               totalTime,
               totalTime / static_cast<double>(mSimulationProps.getNumIterations()));
  }
}

void HeatSolverBase::storeDataIntoFile(hid_t fileHandle, std::size_t iteration, const float *data)
{
  if (fileHandle == H5I_INVALID_HID)
  {
    return;
  }

  std::array gridSize{static_cast<hsize_t>(mMaterialProps.getEdgeSize()),
                      static_cast<hsize_t>(mMaterialProps.getEdgeSize())};

  // Create new HDF5 file group named as "Timestep_N", where "N" is number
  // of current snapshot. The group is placed into root of the file "/Timestep_N".
  std::string groupName = "Timestep_" + std::to_string(iteration / mSimulationProps.getWriteIntensity());
  Hdf5GroupHandle groupHandle(H5Gcreate(fileHandle, groupName.c_str(),
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  {
    // Create new dataset "/Timestep_N/Temperature" which is simulation-domain sized 2D array of "float"s.
    static constexpr std::string_view dataSetName{"Temperature"};
    // Define shape of the dataset (2D edgeSize x edgeSize array).
    Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
    // Create datased with specified shape.
    Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                              H5T_NATIVE_FLOAT, dataSpaceHandle,
                                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    // Write the data from memory pointed by "data" into new datased.
    // Note that we are filling whole dataset and therefore we can specify
    // "H5S_ALL" for both memory and dataset spaces.
    H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    // NOTE: Both dataset and dataspace will be closed here automatically (due to RAII).
  }

  {
    // Create Integer attribute in the same group "/Timestep_N/Time"
    // in which we store number of current simulation iteration.
    static constexpr std::string_view attributeName{"Time"};

    // Dataspace is single value/scalar.
    Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR));

    // Create the attribute in the group as double.
    Hdf5AttributeHandle attributeHandle(H5Acreate2(groupHandle, attributeName.data(),
                                                   H5T_IEEE_F64LE, dataSpaceHandle,
                                                   H5P_DEFAULT, H5P_DEFAULT));

    // Write value into the attribute.
    double snapshotTime = double(iteration);
    H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);

    // NOTE: Both dataspace and attribute handles will be released here.
  }

  // NOTE: The group handle will be released here.
}

void HeatSolverBase::updateTile(const float* oldTemp,
                                float*       newTemp,
                                const float* params,
                                const int*   map,
                                std::size_t  offsetX,
                                std::size_t  offsetY,
                                std::size_t  sizeX,
                                std::size_t  sizeY,
                                std::size_t  strideX) const
{
  const float airflowRate = mSimulationProps.getAirflowRate();
  const float coolerTemp  = mMaterialProps.getCoolerTemperature();

# pragma omp parallel for firstprivate(oldTemp, newTemp, params, map, offsetX, offsetY, sizeX, sizeY, strideX, \
                                       airflowRate, coolerTemp)
  for(std::size_t i = offsetY; i < offsetY + sizeY; ++i)
  {
#   pragma omp simd aligned(oldTemp, newTemp, params, map: AlignedAllocator<>::alignment)
    for(std::size_t j = offsetX; j < offsetX + sizeX; ++j)
    {
      computePoint(oldTemp, newTemp,
                   params,
                   map,
                   i, j, strideX,
                   airflowRate,
                   coolerTemp);
    }
  }
}
