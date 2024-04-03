/**
 * @file    main.cpp
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2022-02-03
 */

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <mpi.h>

#include "AlignedAllocator.hpp"
#include "HeatSolverBase.hpp"
#include "MaterialProperties.hpp"
#include "ParallelHeatSolver.hpp"
#include "SequentialHeatSolver.hpp"
#include "SimulationProperties.hpp"
#include "utils.hpp"

void waitForDebugStart()
{
  using namespace std::chrono_literals;

  bool debugStarted{false};

  while (!debugStarted)
  {
    std::this_thread::sleep_for(50ms);
  }
}

int main(int argc, char *argv[])
{
#ifdef _OPENMP
  int providedParallelism{};

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedParallelism);

  if(providedParallelism < MPI_THREAD_FUNNELED)
  {
    fmt::print(stderr, "MPI init error, atleast MPI_THREAD_FUNNELED not provided!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
#else
  MPI_Init(&argc, &argv);
#endif /* _OPENMP */

  int rank{};
  int size{};

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try
  {
    SimulationProperties simulationProps{};
    simulationProps.parseCommandLine(argc, argv);

    if (simulationProps.isDebug())
    {
      waitForDebugStart();
    }

    MaterialProperties materialProps{};
    materialProps.load(simulationProps.getMaterialFileName(), rank == 0);

    if(rank == 0)
    {
      simulationProps.printParameters(materialProps);
    }

    std::vector<float, AlignedAllocator<float>> sequentialResult{};
    std::vector<float, AlignedAllocator<float>> parallelResult{};

    if(simulationProps.isRunSequential() && rank == 0)
    {
      if(!simulationProps.isBatchMode())
      {
        fmt::print("============ Running sequential solver =============\n");
      }

      sequentialResult.resize(materialProps.getGridPointCount());
      SequentialHeatSolver heatSolver(simulationProps, materialProps);
      heatSolver.run(sequentialResult);
    }

    if(simulationProps.isRunParallel())
    {
      if(rank == 0)
      {
        if(!simulationProps.isBatchMode())
        {
          fmt::print("============= Running parallel solver ==============");
        }

        parallelResult.resize(materialProps.getGridPointCount());
      }

      ParallelHeatSolver heatSolver(simulationProps, materialProps);
      heatSolver.run(parallelResult);
    }

    if(simulationProps.isValidation() && rank == 0)
    {
      if(simulationProps.isDebug())
      {            
        if(!simulationProps.getDebugImageFileName().empty())
        {
          const std::string seqImageName = SimulationProperties::appendFileNameExt(
                simulationProps.getDebugImageFileName(), "seq", ".png");
          const std::string parImageName = SimulationProperties::appendFileNameExt(
                simulationProps.getDebugImageFileName(), "par", ".png");

          saveAsImage(seqImageName, sequentialResult.data(), materialProps.getEdgeSize());
          saveAsImage(parImageName, parallelResult.data(), materialProps.getEdgeSize());
        }
        else
        {
          fmt::print("=============== Sequential results =================\n");
          printArray2d(sequentialResult.data(), materialProps.getEdgeSize());
          fmt::print("\n");

          fmt::print("================ Parallel results ==================\n");
          printArray2d(parallelResult.data(), materialProps.getEdgeSize());
          fmt::print("\n");
        }
      }

      std::vector<float, AlignedAllocator<float>> absError(materialProps.getGridPointCount(), 0.f);

      auto [ok, errorInfo] = verifyResults(parallelResult.data(),
                                           sequentialResult.data(),
                                           materialProps.getGridPointCount(),
                                           absError.data(),
                                           0.001f);

      if(!ok)
      {
        fmt::print("Maximum error of {} is at [{}, {}]\n", errorInfo.maxError,
                                                           errorInfo.maxErrorIdx / materialProps.getEdgeSize(),
                                                           errorInfo.maxErrorIdx % materialProps.getEdgeSize());
        fmt::print("Verification FAILED\n");
      }
      else
      {
        fmt::print("Max deviation is: {}\n", errorInfo.maxError);
        fmt::print("Verification OK\n");
      }

      if(!simulationProps.getDebugImageFileName().empty())
      {
        const std::string errImageName = SimulationProperties::appendFileNameExt(
              simulationProps.getDebugImageFileName(), "abs_diff", ".png");
              
        saveAsImage(errImageName, absError.data(), materialProps.getEdgeSize(), std::make_pair(0.0f, 0.001f));
      }
    }
  }
  catch (const std::exception& e)
  {
    fmt::print(stderr, "Rank #{} trew an exception: {}\n", rank, e.what());
    fmt::print(stderr, "Aborting application...\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
}
