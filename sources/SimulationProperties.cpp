/**
 * @file    SimulationProperties.cpp
 * 
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *          Kristian Kadlubiak <ikadlubiak@fit.vutbr.cz>
 *          David Bayer <ibayer@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          Parameters of the simulation passed as application arguments.
 *
 * @date    2024-02-22
 */

#include <cmath>
#include <iostream>
#include <string>
#include <map>

#ifdef _OPENMP
# include <omp.h>
#endif /* _OPENMP */
#include <mpi.h>

#include <cxxopts.hpp>
#include <fmt/format.h>

#include "SimulationProperties.hpp"

SimulationProperties::SimulationProperties()
: mNIterations(1),
  mWriteIntensity(1000),
  mAirflowRate(0.001f),
  mExecution(Execution::seq),
  mDebugFlag(false),
  mVerificationFlag(false),
  mSequentialFlag(false),
  mBatchMode(false),
  mBatchModeHeader(false),
  mUseParallelIO(false),
  mDecomposition(Decomposition::d1)
{}

void SimulationProperties::parseCommandLine(int argc, char *argv[])
{
  cxxopts::Options options("ppp_proj01", "Heat diffustion simulation with MPI and OpenMP");

  options.add_options()
    ("m,mode", "Simulation mode\n"
               "  0 - run sequential version\n"
               "  1 - run parallel version point-to-point\n"
               "  2 - run parallel version RMA", cxxopts::value<int>(), "<int>")
    ("n,iterations", "Number of iterations", cxxopts::value<std::size_t>(), "<uint>")
    ("i,input", "Input material file name", cxxopts::value<std::string>(), "file")
    ("t,threads", "Number of OpenMP threads", cxxopts::value<int>()->default_value("1"), "<uint>")
    ("o,output", "Output HDF5 file name", cxxopts::value<std::string>()->default_value(""), "<string>")
    ("w,write-intensity", "Disk write intensity", cxxopts::value<std::size_t>()->default_value("50"), "<uint>")
    ("a,airflow-rate", "Air flow rate (values in <0.0001, 0.5> make sense)",
                       cxxopts::value<float>()->default_value("0.001f"), "<float>")
    ("d,debug", "Enable debugging (copare results of SEQ and PAR versions)")
    ("v,verify", "Verification mode (compare results of SEQ and PAR versions)")
    ("b,batch", "Batch mode")
    ("B,batch-with-header", "Batch mode with header")
    ("p,parallel-io", "Parallel I/O mode")
    ("r,render", "Render results into *.png image", cxxopts::value<std::string>(), "<string>")
    ("g,decomp-2d", "Use 2D decomposition instead of 1D")
    ("h,help", "Print usage");    

  try
  {
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      int rank{};

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if(rank == 0)
      {
        fmt::print("{}\n", options.help());
      }

      MPI_Abort(MPI_COMM_WORLD, 0);
    }

    mNIterations      = result["iterations"].as<std::size_t>();
    mWriteIntensity   = result["write-intensity"].as<std::size_t>();
    mAirflowRate      = result["airflow-rate"].as<float>();
    mDebugFlag        = result["debug"].as<bool>();
    mExecution        = static_cast<Execution>(result["mode"].as<int>());
    mVerificationFlag = result["verify"].as<bool>();
    mMaterialFileName = result["input"].as<std::string>();
    mOutputFileName   = result["output"].as<std::string>();
    mBatchMode        = result["batch"].as<bool>() || result["batch-with-header"].as<bool>();
    mBatchModeHeader  = result["batch-with-header"].as<bool>();
    mUseParallelIO    = result["parallel-io"].as<bool>();
    mThreadCount      = result["threads"].as<int>();
    mDebugImageName   = result.count("render") ? result["render"].as<std::string>() : "";
    mDecomposition    = result["decomp-2d"].as<bool>() ? Decomposition::d2 : Decomposition::d1;

    switch (mExecution)
    {
    case Execution::seq:
    case Execution::par_p2p:
    case Execution::par_rma:
      break;
    default:
      throw std::invalid_argument("Invalid mode value");
    }
  }
  catch(const std::exception& e)
  {
    int rank{};

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0)
    {
      std::cerr << "Arguments parse error: " << e.what() << std::endl << std::endl;
      std::cerr << options.help() << std::endl;
    }

    MPI_Abort(MPI_COMM_WORLD, 1);
  }    

#ifdef _OPENMP
  omp_set_num_threads(mThreadCount);
#endif /* _OPENMP */

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Compute how arrangement of the processes into the grid
  if(getDecomposition() == SimulationProperties::Decomposition::d1)
  {
    mGridSize[0] = size;
    mGridSize[1] = 1;
  }
  else
  {
    bool isEvenPower = (static_cast<int>(std::log2(size)) % 2 == 0);

    mGridSize[0] = static_cast<int>(sqrt(size / (isEvenPower ? 1 : 2)));
    mGridSize[1] = mGridSize[0] * (isEvenPower ? 1 : 2);
  }
}

void SimulationProperties::printParameters(const MaterialProperties &materialProps) const
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(mBatchMode)
  {
    if(mBatchModeHeader)
    {
      fmt::print("mpi_procs;"
                 "grid_tiles_x;"
                 "grid_tiles_y;"
                 "omp_threads;"
                 "domain_size;"
                 "n_iterations;"
                 "disk_write_intensity;"
                 "airflow;"
                 "material_file;"
                 "output_file;"
                 "simulation_mode;"
                 "middle_col_avg_temp;"
                 "total_time;"
                 "iteration_time\n");

    }

    fmt::print("{};{};{};{};{};{};{};{};{};",
               size,
               mGridSize[0], mGridSize[1],
               getThreadCount(),
               materialProps.getEdgeSize(),
               getNumIterations(),
               getWriteIntensity(),
               getAirflowRate(),
               getMaterialFileName());
  }
  else
  {
    fmt::print(".......... Parameters of the simulation ...........\n"
               "Domain size:              {}x{}\n"
               "Number of iterations:     {}\n"
               "Number of MPI processes:  {}\n"
               "Number of OpenMP threads: {}\n"
               "Disk write intensity:     {}\n"
               "Air flow rate:            {}\n"
               "Input file name:          {}\n"
               "Output file name:         {}\n"
               "Execution:                {}\n"
               "Decomposition type:       {} ({}, {})\n"
               "...................................................\n",
               materialProps.getEdgeSize(), materialProps.getEdgeSize(),
               getNumIterations(),
               size,
               getThreadCount(),
               getWriteIntensity(),
               getAirflowRate(),
               getMaterialFileName(),
               getOutputFileName(),
               static_cast<std::underlying_type_t<Execution>>(mExecution),
               ((mDecomposition == Decomposition::d1) ? "1D" : "2D"),
               mGridSize[0], mGridSize[1]);
  }
}

bool SimulationProperties::isRunSequential() const
{
  return (mVerificationFlag || mDebugFlag || mExecution == Execution::seq);
}

bool SimulationProperties::isRunParallel() const
{
  return (mVerificationFlag || mDebugFlag || mExecution == Execution::par_p2p || mExecution == Execution::par_rma);
}

bool SimulationProperties::isRunParallelP2P() const
{
  return (mExecution == Execution::par_p2p);
}

bool SimulationProperties::isRunParallelRMA() const
{
  return (mExecution == Execution::par_rma);
}

bool SimulationProperties::isValidation() const
{
  return (mVerificationFlag || mDebugFlag);
}

std::size_t SimulationProperties::getNumIterations() const
{
  return mNIterations;
}

int SimulationProperties::getThreadCount() const
{
  return mThreadCount;
}

std::size_t SimulationProperties::getWriteIntensity() const
{
  return mWriteIntensity;
}

float SimulationProperties::getAirflowRate() const
{
  return mAirflowRate;
}

std::string_view SimulationProperties::getMaterialFileName() const
{
  return mMaterialFileName;
}

std::string SimulationProperties::getOutputFileName(std::string_view codeTypeExt) const
{
  if(mOutputFileName.empty())
  {
    return std::string{};
  }
  else if(codeTypeExt.empty())
  {
    return mOutputFileName;
  }
  else
  {
    return appendFileNameExt(mOutputFileName, codeTypeExt, ".h5");
  }
}

std::string_view SimulationProperties::getDebugImageFileName() const
{
  return mDebugImageName;
}

bool SimulationProperties::isBatchMode() const
{
  return mBatchMode;
}

bool SimulationProperties::useParallelIO() const
{
  return mUseParallelIO;
}

bool SimulationProperties::isDebug() const
{
  return mDebugFlag;
}

SimulationProperties::Decomposition SimulationProperties::getDecomposition() const
{
  return mDecomposition;
}

void SimulationProperties::getDecompGrid(int& outSizeX, int& outSizeY) const
{
  outSizeX = mGridSize[0];
  outSizeY = mGridSize[1];
}

std::string SimulationProperties::appendFileNameExt(std::string_view fileName,
                                                    std::string_view fileNameExt,
                                                    std::string_view fileTypeExt)
{
  using namespace std::literals;

  std::string result{fileName};

  if(fileName.find(fileTypeExt) == std::string::npos)
  {
    result.append("_");
    result.append(fileNameExt);
    result.append(fileTypeExt);
  }
  else
  {
    result.insert(result.find_last_of("."), "_"s.append(fileNameExt));
  }

  return result;
}
