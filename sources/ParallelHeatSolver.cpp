/**
 * @file    ParallelHeatSolver.cpp
 * 
 * @author  Name Surname <xlogin00@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <ios>
#include <string_view>

#include "ParallelHeatSolver.hpp"

ParallelHeatSolver::ParallelHeatSolver(const SimulationProperties& simulationProps,
                                       const MaterialProperties&   materialProps)
: HeatSolverBase(simulationProps, materialProps) {
  MPI_Comm_size(MPI_COMM_WORLD, &mWorldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mWorldRank);

/**********************************************************************************************************************/
/*                                  Call init* and alloc* methods in correct order                                    */
/**********************************************************************************************************************/
  initGridTopology();
  initDataDistribution();
  allocLocalTiles();

  if(!mSimulationProps.getOutputFileName().empty()) {
/**********************************************************************************************************************/
/*                               Open output file if output file name was specified.                                  */
/*  If mSimulationProps.useParallelIO() flag is set to true, open output file for parallel access, otherwise open it  */
/*                         only on MASTER rank using sequetial IO. Use openOutputFile* methods.                       */
/**********************************************************************************************************************/


  }
}

ParallelHeatSolver::~ParallelHeatSolver() {
/**********************************************************************************************************************/
/*                                  Call deinit* and dealloc* methods in correct order                                */
/*                                             (should be in reverse order)                                           */
/**********************************************************************************************************************/
  deallocLocalTiles();
  deinitDataDistribution();
}

std::string_view ParallelHeatSolver::getCodeType() const
{
  return codeType;
}


void ParallelHeatSolver::initGridTopology() {
/**********************************************************************************************************************/
/*                          Initialize 2D grid topology using non-periodic MPI Cartesian topology.                    */
/*                       Also create a communicator for middle column average temperature computation.                */
/**********************************************************************************************************************/
  int nx, ny;
  mSimulationProps.getDecompGrid(nx, ny);
  total_size = nx * ny;

  dims.resize(2);
  std::vector<int> periods = {0, 0};
  switch (mSimulationProps.getDecomposition()) {
    case SimulationProperties::Decomposition::d1:
      dims[0] = nx; dims[1] = ny;
      break;
    // case SimulationProperties::Decomposition::d2:
    //   if (nx == ny) {
    //     std::array dims{nx, ny};
    //     std::array periods{0, 0};
    //   } else {
    //   }
    //   break;
    default:
      break;
  }
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1, &cart_comm);
  cart_coords.resize(2);
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Cart_coords(cart_comm, cart_rank, 2, cart_coords.data());
  // calculate neighbors
  neighbors.resize(8);
  MPI_Cart_shift(cart_comm, 0, 1, &neighbors[ND::L], &neighbors[ND::R]);
  MPI_Cart_shift(cart_comm, 0, 2, &neighbors[ND::LL], &neighbors[ND::RR]);
  MPI_Cart_shift(cart_comm, 1, 1, &neighbors[ND::U], &neighbors[ND::D]);
  MPI_Cart_shift(cart_comm, 1, 2, &neighbors[ND::UU], &neighbors[ND::DD]);
  
  // TODO: create communicator for parallel calculation of the average temp
}

void ParallelHeatSolver::deinitGridTopology() {
/**********************************************************************************************************************/
/*      Deinitialize 2D grid topology and the middle column average temperature computation communicator              */
/**********************************************************************************************************************/
  MPI_Comm_free(&cart_comm);
  MPI_Comm_free(&center_col);
}

void ParallelHeatSolver::initDataDistribution() {
/**********************************************************************************************************************/
/*                 Initialize variables and MPI datatypes for data distribution (float and int).                      */
/**********************************************************************************************************************/
  // global tile size
  global_edge_size = mMaterialProps.getEdgeSize();
  // calculate the local tile size
  tile_size_x = global_edge_size / dims[0];
  tile_size_y = global_edge_size / dims[1];

  // only root needs the tile type
  if (mWorldRank == 0) {
    std::array domain_dims = {global_edge_size, global_edge_size};
    std::array tile_dims = {tile_size_x, tile_size_y};
    std::array start_arr = {0, 0};
    
    MPI_Datatype tile_org_type{MPI_DATATYPE_NULL};

    MPI_Type_create_subarray(2, domain_dims.data(), tile_dims.data(), start_arr.data(), MPI_ORDER_C, MPI_FLOAT, &tile_org_type);
    MPI_Type_create_resized(tile_org_type, 0, 1 * sizeof(float), &tile_type);
    MPI_Type_commit(&tile_type);
  }

  // counts = nx * ny = world_size = n_processors
  counts = std::make_unique<int[]>(total_size);
  displacements = std::make_unique<int[]>(total_size);
  std::fill_n(counts.get(), total_size, 1);
  
  for (int i=0; i < total_size; i++) {
      displacements[i] = i * tile_size_x * tile_size_y + ((i * tile_size_x) % global_edge_size);
  }

  // halo row int
  MPI_Type_contiguous(tile_size_x * 2, MPI_INT, &local_row_int);
  MPI_Type_commit(&local_row_int);
  
  // halo row float
  MPI_Type_contiguous(tile_size_x * 2, MPI_FLOAT, &local_row_float);
  MPI_Type_commit(&local_row_float);

  // halo column int
  MPI_Type_vector(tile_size_y, 2, tile_size_x, MPI_INT, &local_col_int);
  MPI_Type_commit(&local_col_int);

  // halo column float
  MPI_Type_vector(tile_size_y, 2, tile_size_x, MPI_INT, &local_col_float);
  MPI_Type_commit(&local_col_float);
}

void ParallelHeatSolver::deinitDataDistribution() {
/**********************************************************************************************************************/
/*                       Deinitialize variables and MPI datatypes for data distribution.                              */
/**********************************************************************************************************************/


}

void ParallelHeatSolver::allocLocalTiles() {
/**********************************************************************************************************************/
/*            Allocate local tiles for domain map (1x), domain parameters (1x) and domain temperature (2x).           */
/*                                               Use AlignedAllocator.                                                */
/**********************************************************************************************************************/


}

void ParallelHeatSolver::deallocLocalTiles() {
/**********************************************************************************************************************/
/*                                   Deallocate local tiles (may be empty).                                           */
/**********************************************************************************************************************/


}

void ParallelHeatSolver::initHaloExchange()
{
/**********************************************************************************************************************/
/*                            Initialize variables and MPI datatypes for halo exchange.                               */
/*                    If mSimulationProps.isRunParallelRMA() flag is set to true, create RMA windows.                 */
/**********************************************************************************************************************/
  if (mSimulationProps.isRunParallelP2P()) {

  } else if(mSimulationProps.isRunParallelRMA()) {

  }
}

void ParallelHeatSolver::deinitHaloExchange()
{
/**********************************************************************************************************************/
/*                            Deinitialize variables and MPI datatypes for halo exchange.                             */
/**********************************************************************************************************************/
  
  
}

template<typename T>
void ParallelHeatSolver::scatterTiles(const T* globalData, T* localData)
{
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported scatter datatype!");

/**********************************************************************************************************************/
/*                      Implement master's global tile scatter to each rank's local tile.                             */
/*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
/*                                                                                                                    */
/*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
/*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
/**********************************************************************************************************************/

  
}

template<typename T>
void ParallelHeatSolver::gatherTiles(const T* localData, T* globalData)
{
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported gather datatype!");

/**********************************************************************************************************************/
/*                      Implement each rank's local tile gather to master's rank global tile.                         */
/*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
/*                                                                                                                    */
/*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
/*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
/**********************************************************************************************************************/


}

void ParallelHeatSolver::computeHaloZones(const float* oldTemp, float* newTemp)
{
/**********************************************************************************************************************/
/*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
/*                        Use updateTile method to compute new temperatures in halo zones.                            */
/*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
/**********************************************************************************************************************/
  
}

void ParallelHeatSolver::startHaloExchangeP2P(float* localData, std::array<MPI_Request, 8>& requests)
{
/**********************************************************************************************************************/
/*                       Start the non-blocking halo zones exchange using P2P communication.                          */
/*                         Use the requests array to return the requests from the function.                           */
/*                            Don't forget to set the empty requests to MPI_REQUEST_NULL.                             */
/**********************************************************************************************************************/


}

void ParallelHeatSolver::startHaloExchangeRMA(float* localData, MPI_Win window)
{
/**********************************************************************************************************************/
/*                       Start the non-blocking halo zones exchange using RMA communication.                          */
/*                   Do not forget that you put/get the values to/from the target's opposite side                     */
/**********************************************************************************************************************/
  
  
}

void ParallelHeatSolver::awaitHaloExchangeP2P(std::array<MPI_Request, 8>& requests)
{
/**********************************************************************************************************************/
/*                       Wait for all halo zone exchanges to finalize using P2P communication.                        */
/**********************************************************************************************************************/

  
}

void ParallelHeatSolver::awaitHaloExchangeRMA(MPI_Win window)
{
/**********************************************************************************************************************/
/*                       Wait for all halo zone exchanges to finalize using RMA communication.                        */
/**********************************************************************************************************************/


}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>>& outResult)
{
  std::array<MPI_Request, 8> requestsP2P{};

/**********************************************************************************************************************/
/*                                         Scatter initial data.                                                      */
/**********************************************************************************************************************/



/**********************************************************************************************************************/
/* Exchange halo zones of initial domain temperature and parameters using P2P communication. Wait for them to finish. */
/**********************************************************************************************************************/



/**********************************************************************************************************************/
/*                            Copy initial temperature to the second buffer.                                          */
/**********************************************************************************************************************/



  double startTime = MPI_Wtime();

  // 3. Start main iterative simulation loop.
  for(std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter)
  {
    const std::size_t oldIdx = iter % 2;       // Index of the buffer with old temperatures
    const std::size_t newIdx = (iter + 1) % 2; // Index of the buffer with new temperatures

/**********************************************************************************************************************/
/*                            Compute and exchange halo zones using P2P or RMA.                                       */
/**********************************************************************************************************************/

    

/**********************************************************************************************************************/
/*                           Compute the rest of the tile. Use updateTile method.                                     */
/**********************************************************************************************************************/



/**********************************************************************************************************************/
/*                            Wait for all halo zone exchanges to finalize.                                           */
/**********************************************************************************************************************/



    if(shouldStoreData(iter))
    {
/**********************************************************************************************************************/
/*                          Store the data into the output file using parallel or sequential IO.                      */
/**********************************************************************************************************************/


    }

    if(shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature())
    {
/**********************************************************************************************************************/
/*                 Compute and print middle column average temperature and print progress report.                     */
/**********************************************************************************************************************/


    }
  }

  const std::size_t resIdx = mSimulationProps.getNumIterations() % 2; // Index of the buffer with final temperatures

  double elapsedTime = MPI_Wtime() - startTime;

/**********************************************************************************************************************/
/*                                     Gather final domain temperature.                                               */
/**********************************************************************************************************************/



/**********************************************************************************************************************/
/*           Compute (sequentially) and report final middle column temperature average and print final report.        */
/**********************************************************************************************************************/


}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const
{
/**********************************************************************************************************************/
/*                Return true if rank should compute middle column average temperature.                               */
/**********************************************************************************************************************/

  return false;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureParallel(const float *localData) const
{
/**********************************************************************************************************************/
/*                  Implement parallel middle column average temperature computation.                                 */
/*                      Use OpenMP directives to accelerate the local computations.                                   */
/**********************************************************************************************************************/
  
  

  return 0.f;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureSequential(const float *globalData) const
{  
/**********************************************************************************************************************/
/*                  Implement sequential middle column average temperature computation.                               */
/*                      Use OpenMP directives to accelerate the local computations.                                   */
/**********************************************************************************************************************/

  

  return 0.f;
}

void ParallelHeatSolver::openOutputFileSequential()
{
  // Create the output file for sequential access.
  mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                          H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(!mFileHandle.valid())
  {
    throw std::ios::failure("Cannot create output file!");
  }
}

void ParallelHeatSolver::storeDataIntoFileSequential(hid_t        fileHandle,
                                                     std::size_t  iteration,
                                                     const float* globalData)
{
  storeDataIntoFile(fileHandle, iteration, globalData);
}

void ParallelHeatSolver::openOutputFileParallel()
{
#ifdef H5_HAVE_PARALLEL
  Hdf5PropertyListHandle faplHandle{};

/**********************************************************************************************************************/
/*                          Open output HDF5 file for parallel access with alignment.                                 */
/*      Set up faplHandle to use MPI-IO and alignment. The handle will automatically release the resource.            */
/**********************************************************************************************************************/
  


  mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                          H5F_ACC_TRUNC,
                          H5P_DEFAULT,
                          faplHandle);
  if(!mFileHandle.valid())
  {
    throw std::ios::failure("Cannot create output file!");
  }
#else
  throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}

void ParallelHeatSolver::storeDataIntoFileParallel(hid_t                         fileHandle,
                                                   [[maybe_unused]] std::size_t  iteration,
                                                   [[maybe_unused]] const float* localData)
{
  if (fileHandle == H5I_INVALID_HID)
  {
    return;
  }

#ifdef H5_HAVE_PARALLEL
  std::array gridSize{static_cast<hsize_t>(mMaterialProps.getEdgeSize()),
                      static_cast<hsize_t>(mMaterialProps.getEdgeSize())};

  // Create new HDF5 group in the output file
  std::string groupName = "Timestep_" + std::to_string(iteration / mSimulationProps.getWriteIntensity());

  Hdf5GroupHandle groupHandle(H5Gcreate(fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  {
/**********************************************************************************************************************/
/*                                Compute the tile offsets and sizes.                                                 */
/*               Note that the X and Y coordinates are swapped (but data not altered).                                */
/**********************************************************************************************************************/
    
    

    // Create new dataspace and dataset using it.
    static constexpr std::string_view dataSetName{"Temperature"};

    Hdf5PropertyListHandle datasetPropListHandle{};

/**********************************************************************************************************************/
/*                            Create dataset property list to set up chunking.                                        */
/*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
/**********************************************************************************************************************/



    Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
    Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                              H5T_NATIVE_FLOAT, dataSpaceHandle,
                                              H5P_DEFAULT, datasetPropListHandle,
                                              H5P_DEFAULT));

    Hdf5DataspaceHandle memSpaceHandle{};

/**********************************************************************************************************************/
/*                Create memory dataspace representing tile in the memory (set up memSpaceHandle).                    */
/**********************************************************************************************************************/



/**********************************************************************************************************************/
/*              Select inner part of the tile in memory and matching part of the dataset in the file                  */
/*                           (given by position of the tile in global domain).                                        */
/**********************************************************************************************************************/



    Hdf5PropertyListHandle propListHandle{};

/**********************************************************************************************************************/
/*              Perform collective write operation, writting tiles from all processes at once.                        */
/*                                   Set up the propListHandle variable.                                              */
/**********************************************************************************************************************/
    
    

    H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, propListHandle, localData);
  }

  {
    // 3. Store attribute with current iteration number in the group.
    static constexpr std::string_view attributeName{"Time"};
    Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR));
    Hdf5AttributeHandle attributeHandle(H5Acreate2(groupHandle, attributeName.data(),
                                                   H5T_IEEE_F64LE, dataSpaceHandle,
                                                   H5P_DEFAULT, H5P_DEFAULT));
    const double snapshotTime = static_cast<double>(iteration);
    H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
  }
#else
  throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}
