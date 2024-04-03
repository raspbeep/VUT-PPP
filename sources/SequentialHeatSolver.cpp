/**
 * @file    SequentialHeatSolver.cpp
 * 
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *          David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of sequential heat equation solver.
 *
 * @date    2024-02-23
 */

#include <array>
#include <ios>

#include "SequentialHeatSolver.hpp"

SequentialHeatSolver::SequentialHeatSolver(const SimulationProperties& simulationProps,
                                           const MaterialProperties&   materialProps)
: HeatSolverBase(simulationProps, materialProps),
  mTempArray(materialProps.getGridPointCount()),
  mFileHandle()
{
  // 1. Open output file if its name was specified.
  if(!mSimulationProps.getOutputFileName().empty())
  {
    const std::string outputFileName = mSimulationProps.getOutputFileName(codeType);

    mFileHandle = H5Fcreate(outputFileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    if (!mFileHandle.valid())
    {
      throw std::ios::failure("Cannot create output file");
    }
  }
}

void SequentialHeatSolver::run(std::vector<float, AlignedAllocator<float>>& outResult)
{
  // 2. Copy initial temperature into both working arrays
  std::copy(mMaterialProps.getInitialTemperature().begin(),
            mMaterialProps.getInitialTemperature().end(),
            mTempArray.begin());

  std::copy(mMaterialProps.getInitialTemperature().begin(),
            mMaterialProps.getInitialTemperature().end(),
            outResult.begin());

  std::array workTempArrays{mTempArray.data(), outResult.data()};
  float      middleColAvgTemp{};

  const double startTime = MPI_Wtime();

  const float airflowRate = mSimulationProps.getAirflowRate();
  const float coolerTemp  = mMaterialProps.getCoolerTemperature();

  // 3. Begin iterative simulation main loop
  for(std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter)
  {
    // 4. Compute new temperature for each point in the domain (except borders)
    // border temperatures should remain constant (plus our stencil is +/-2 points).
    for(std::size_t i = 2; i < mMaterialProps.getEdgeSize() - 2; ++i)
    {
      for(std::size_t j = 2; j < mMaterialProps.getEdgeSize() - 2; ++j)
      {
        computePoint(workTempArrays[1], workTempArrays[0],
                     mMaterialProps.getDomainParameters().data(),
                     mMaterialProps.getDomainMap().data(),
                     i, j,
                     mMaterialProps.getEdgeSize(),
                     airflowRate,
                     coolerTemp);
      }
    }

    // 5. Compute average temperature in the middle column of the domain.
    middleColAvgTemp = computeMiddleColAvgTemp(workTempArrays[0]);

    // 6. Store the simulation state if appropriate (ie. every N-th iteration)
    if(shouldStoreData(iter))
    {
      storeDataIntoFile(mFileHandle, iter, workTempArrays[0]);
    }

    // 7. Swap source and destination buffers
    std::swap(workTempArrays[0], workTempArrays[1]);

    // 8. Print current progress (prints progress only every 10% of the simulation).
    printProgressReport(iter, middleColAvgTemp);
  }

  // 9. Measure total execution time and report
  const double elapsedTime = MPI_Wtime() - startTime;
  printFinalReport(elapsedTime, middleColAvgTemp);

  // 10. Copy result over if necessary (even/odd number of buffer swaps).
  if(mSimulationProps.getNumIterations() & 1)
  {
    std::copy(mTempArray.begin(), mTempArray.end(), outResult.begin());
  }
}

std::string_view SequentialHeatSolver::getCodeType() const
{
  return codeType;
}

float SequentialHeatSolver::computeMiddleColAvgTemp(const float *data) const
{
  float middleColAvgTemp{};

  for(std::size_t i = 0; i < mMaterialProps.getEdgeSize(); ++i)
  {
    middleColAvgTemp += data[i * mMaterialProps.getEdgeSize() + mMaterialProps.getEdgeSize() / 2];
  }

  return middleColAvgTemp / static_cast<float>(mMaterialProps.getEdgeSize());
}
