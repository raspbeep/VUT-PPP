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
    std::cerr << "MPI init error, atleast MPI_THREAD_FUNNELED not provided!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
#else
  MPI_Init(&argc, &argv);
#endif /* _OPENMP */
  printf("here\n");

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
        std::cout << "============ Running sequential solver =============" << std::endl;
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
          std::cout << "============= Running parallel solver ==============" << std::endl;
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
          std::cout << "=============== Sequential results =================" << std::endl;
          printArray2d(sequentialResult.data(), materialProps.getEdgeSize());
          std::cout << std::endl;

          std::cout << "================ Parallel results ==================" << std::endl;
          printArray2d(parallelResult.data(), materialProps.getEdgeSize());
          std::cout << std::endl;
        }
      }

      std::vector<float, AlignedAllocator<float>> absError(materialProps.getGridPointCount(), 0.f);

      auto [ok, errorInfo] = verifyResults(parallelResult.data(),
                                           sequentialResult.data(),
                                           materialProps.getGridPointCount(),
                                           absError.data(),
                                           0.001f);

      if(ok)
      {
        std::cout << "Maximum error of " << std::scientific
                  << errorInfo.maxError << std::defaultfloat
                  << " is at [" << (errorInfo.maxErrorIdx / materialProps.getEdgeSize()) << ", "
                  << (errorInfo.maxErrorIdx % materialProps.getEdgeSize()) << "]\n"
                  << "Verification FAILED" << std::endl;
      }
      else
      {
        std::cout << "Max deviation is: " << std::scientific
                  << errorInfo.maxError << std::defaultfloat << std::endl
                  << "Verification OK" << std::endl;
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
    std::cerr << "Rank #" << rank << " trew an exception: " << e.what() << "\nAborting application..." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
}
