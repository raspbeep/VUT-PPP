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
 * 
 * @question Ktore vsetky ranky pustaju konstruktor? Je to iba root? Treba osetrovat ci sa napriklad datatype na celu domenu inicializuje iba na root ranku?
 *  
 * @question Co sa ma diat vo funkciach ako deallocLocalTiles? AFAIK su tam iba arrays ktore sa dealokuju podla Allocator, ktory by to mal robit automaticky.
 * @question Preco ma updateTile() v komentari minimalny offset >=2?
 * @question V computeHaloZones(), je tam "Take care not to compute some regions twice" kvoli efektivite alebo konzistencii?
 * @question Ktory je center column celej domeny? Ak mame edgeSize vzdy mocninu 2, tak je to lavy alebo pravy?
 * @question V startHaloExchangeP2P(), ocakava sa, ze bude mat localData rozmery (tile_size_x + haloSize) x (tile_size_y + haloSize)?
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

  std::vector<int> periods = {0, 0};
  switch (mSimulationProps.getDecomposition()) {
    case SimulationProperties::Decomposition::d1:
      n_dims = 1;
      dims[0] = nx; dims[1] = 0;
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
  MPI_Cart_create(MPI_COMM_WORLD, n_dims, dims.data(), periods.data(), 1, &cart_comm);
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Cart_coords(cart_comm, cart_rank, n_dims, cart_coords.data());
  
  // calculate neighbors, MPI automatically assigns MPI_PROC_NULL
  MPI_Cart_shift(cart_comm, 0, 1, &neighbors[ND::W], &neighbors[ND::E]);
  MPI_Cart_shift(cart_comm, 1, 1, &neighbors[ND::N], &neighbors[ND::S]);
  // TODO: create communicator for parallel calculation of the average temp
}

void ParallelHeatSolver::deinitGridTopology() {
/**********************************************************************************************************************/
/*      Deinitialize 2D grid topology and the middle column average temperature computation communicator              */
/**********************************************************************************************************************/
  MPI_Comm_free(&cart_comm);
  // TODO: center col
  // MPI_Comm_free(&center_col);
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

  tile_size_with_halo_x = tile_size_x + 2 * haloZoneSize;
  tile_size_with_halo_y = tile_size_y + 2 * haloZoneSize;

  // only root needs the tile type
  if (mWorldRank == 0) {
    std::array domain_dims = {global_edge_size, global_edge_size};
    std::array tile_dims = {tile_size_x, tile_size_y};
    std::array start_arr = {0, 0};
    
    // datatypes to derive the resized types from
    MPI_Datatype tile_org_type_float{MPI_DATATYPE_NULL};
    MPI_Datatype tile_org_type_int{MPI_DATATYPE_NULL};

    // global subarray float type for sending a tile from the root rank
    MPI_Type_create_subarray(2, domain_dims.data(), tile_dims.data(), start_arr.data(), MPI_ORDER_C, MPI_FLOAT, &tile_org_type_float);
    MPI_Type_create_resized(tile_org_type_float, 0, 1 * sizeof(float), &global_tile_type_float);
    MPI_Type_commit(&global_tile_type_float);

    // global subarray int type for sending a tile from the root rank
    MPI_Type_create_subarray(2, domain_dims.data(), tile_dims.data(), start_arr.data(), MPI_ORDER_C, MPI_INT, &tile_org_type_int);
    MPI_Type_create_resized(tile_org_type_int, 0, 1 * sizeof(int), &global_tile_type_int);
    MPI_Type_commit(&global_tile_type_int);

    MPI_Type_free(&tile_org_type_float);
    MPI_Type_free(&tile_org_type_int);
  }

  // tile with a border of two around on the edges
  local_tile_with_halo_dims = {tile_size_with_halo_x, tile_size_with_halo_y};
  // original tile size for receiving
  std::array local_tile_dims = {tile_size_x, tile_size_y};
  std::array<int, 2> start_arr = {haloZoneSize, haloZoneSize};
  
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), local_tile_dims.data(), start_arr.data(), MPI_ORDER_C, MPI_FLOAT, &local_tile_type_float);
  MPI_Type_commit(&local_tile_type_float);

  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), local_tile_dims.data(), start_arr.data(), MPI_ORDER_C, MPI_INT, &local_tile_type_int);
  MPI_Type_commit(&local_tile_type_int);

  // counts = nx * ny = world_size = n_processors
  counts = std::make_unique<int[]>(total_size);
  displacements = std::make_unique<int[]>(total_size);
  std::fill_n(counts.get(), total_size, 1);
  
  for (int i=0; i < total_size; i++) {
      displacements[i] = i * tile_size_x * tile_size_y + ((i * tile_size_x) % global_edge_size);
  }
}

void ParallelHeatSolver::deinitDataDistribution() {
/**********************************************************************************************************************/
/*                       Deinitialize variables and MPI datatypes for data distribution.                              */
/**********************************************************************************************************************/
  
  if(mWorldRank == 0) {
    MPI_Type_free(&global_tile_type_float);
    MPI_Type_free(&global_tile_type_int);
  }

  MPI_Type_free(&local_tile_type_float);
  MPI_Type_free(&local_tile_type_int);
  
  // TODO: what variables?
  // clear and shrink counts and displacements?
}

void ParallelHeatSolver::allocLocalTiles() {
/**********************************************************************************************************************/
/*            Allocate local tiles for domain map (1x), domain parameters (1x) and domain temperature (2x).           */
/*                                               Use AlignedAllocator.                                                */
/**********************************************************************************************************************/
  // TODO: maybe obsolete?
  if (mWorldRank == 0) {
    domain.resize(global_edge_size * global_edge_size);
  }
  
  tile_map.resize(tile_size_x * tile_size_y);
  tile_params.resize(tile_size_x * tile_size_y);

  tile_temps[OLD].resize(tile_size_x * tile_size_y);
  tile_temps[NEW].resize(tile_size_x * tile_size_y);
}

void ParallelHeatSolver::deallocLocalTiles() {
/**********************************************************************************************************************/
/*                                   Deallocate local tiles (may be empty).                                           */
/**********************************************************************************************************************/
  // AlignedAllocator takes care of
}

void ParallelHeatSolver::initHaloExchange() {
/**********************************************************************************************************************/
/*                            Initialize variables and MPI datatypes for halo exchange.                               */
/*                    If mSimulationProps.isRunParallelRMA() flag is set to true, create RMA windows.                 */
/**********************************************************************************************************************/
  if (mSimulationProps.isRunParallelP2P()) {
    // all are subarrays in the local tile of (tile_size_with_halo_x * tile_size_with_halo_y)
    // size of the local tile with halo is local_tile_with_halo_dims

    std::array<int, 2> halo_start_up    = {haloZoneSize, 0};
    std::array<int, 2> halo_start_down  = {haloZoneSize, tile_size_y + haloZoneSize};
    std::array<int, 2> halo_start_left  = {0, haloZoneSize};
    std::array<int, 2> halo_start_right = {tile_size_x + haloZoneSize, haloZoneSize};

    std::array<int, 2> halo_dims_row    =  {tile_size_x, haloZoneSize};
    std::array<int, 2> halo_dims_col    =  {haloZoneSize, tile_size_y};

    // halo row up int
    MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_start_up.data(), MPI_ORDER_C, MPI_INT, &halo_row_up_type_int);
    MPI_Type_commit(&halo_row_up_type_int);

    // halo row down int
    MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_start_down.data(), MPI_ORDER_C, MPI_INT, &halo_row_down_type_int);
    MPI_Type_commit(&halo_row_down_type_int);
    
    // halo row up float
    MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_start_up.data(), MPI_ORDER_C, MPI_FLOAT, &halo_row_up_type_float);
    MPI_Type_commit(&halo_row_up_type_float);

    // halo row down float
    MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_start_down.data(), MPI_ORDER_C, MPI_FLOAT, &halo_row_down_type_float);
    MPI_Type_commit(&halo_row_down_type_float);

    // halo col left int
    MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_start_left.data(), MPI_ORDER_C, MPI_INT, &halo_col_left_type_int);
    MPI_Type_commit(&halo_col_left_type_int);

    // halo col right int
    MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_start_right.data(), MPI_ORDER_C, MPI_INT, &halo_col_right_type_int);
    MPI_Type_commit(&halo_col_right_type_int);
    
    // halo col left float
    MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_start_left.data(), MPI_ORDER_C, MPI_FLOAT, &halo_col_left_type_float);
    MPI_Type_commit(&halo_col_left_type_float);

    // halo col right float
    MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_start_right.data(), MPI_ORDER_C, MPI_FLOAT, &halo_col_right_type_float);
    MPI_Type_commit(&halo_col_right_type_float);

  } else if(mSimulationProps.isRunParallelRMA()) {
    // TODO: RMA
  }
}

void ParallelHeatSolver::deinitHaloExchange() {
/**********************************************************************************************************************/
/*                            Deinitialize variables and MPI datatypes for halo exchange.                             */
/**********************************************************************************************************************/
  if (mSimulationProps.isRunParallelP2P()) {
    MPI_Type_free(&halo_row_up_type_int);
    MPI_Type_free(&halo_row_down_type_int);
    MPI_Type_free(&halo_row_up_type_float);
    MPI_Type_free(&halo_row_down_type_float);

    MPI_Type_free(&halo_col_left_type_int);
    MPI_Type_free(&halo_col_right_type_int);
    MPI_Type_free(&halo_col_left_type_float);
    MPI_Type_free(&halo_col_right_type_float);
  } else if(mSimulationProps.isRunParallelRMA()) {
    // TODO: RMA
  }
}

template<typename T>
void ParallelHeatSolver::scatterTiles(const T* globalData, T* localData) {
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported scatter datatype!");

/**********************************************************************************************************************/
/*                      Implement master's global tile scatter to each rank's local tile.                             */
/*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
/*                                                                                                                    */
/*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
/*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
/**********************************************************************************************************************/
  const MPI_Datatype global_tile_type = std::is_same_v<T, int> ? global_tile_type_int : global_tile_type_float;
  const MPI_Datatype local_tile_type  = std::is_same_v<T, int> ? local_tile_type_int  : local_row_float;
  // TODO: Do I need scatterv?
  MPI_Scatterv(globalData, counts.get(), displacements.get(), global_tile_type, localData, 1, local_tile_type, 0, MPI_COMM_WORLD);
}

template<typename T>
void ParallelHeatSolver::gatherTiles(const T* localData, T* globalData) {
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported gather datatype!");

/**********************************************************************************************************************/
/*                      Implement each rank's local tile gather to master's rank global tile.                         */
/*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
/*                                                                                                                    */
/*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
/*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
/**********************************************************************************************************************/
  const MPI_Datatype global_tile_type = std::is_same_v<T, int> ? global_tile_type_int : global_tile_type_float;
  const MPI_Datatype local_tile_type  = std::is_same_v<T, int> ? local_tile_type_int  : local_tile_type_float;
  
  // TODO: Do I need scatterv?
  MPI_Gatherv(localData, 1, local_tile_type, globalData, counts.get(), displacements.get(), global_tile_type, 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::computeHaloZones(const float* oldTemp, float* newTemp) {
/**********************************************************************************************************************/
/*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
/*                        Use updateTile method to compute new temperatures in halo zones.                            */
/*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
/**********************************************************************************************************************/
  
  // TODO: Possible off by one.
  // in 1D decomp, the dims are (P), but the cart_coords of any tile is (_, 0) and we are checking for the last one
  bool is_top = cart_coords[1] == 0;
  bool is_bottom = cart_coords[1] == dims[1] - 1;
  bool is_left = cart_coords[0] == 0;
  bool is_right = cart_coords[0] == dims[0] - 1;
  bool decomp_1d = mSimulationProps.getDecomposition() == SimulationProperties::Decomposition::d1;
  
  //       ________
  //     __|______|___
  //  __|__|______|__|__
  // |  |  |      |  |  |
  // |  |  |      |  |  |
  // |__|__|______|__|__|
  //    |__|______|__|
  //       |______|

  // in 1d decomp we do not exchange up and bottom halo zones
  if (!is_top && !decomp_1d) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize * 2, haloZoneSize, tile_size_x - (2 * haloZoneSize), haloZoneSize, tile_size_with_halo_x);
  }
  if (!is_bottom && !decomp_1d) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize * 2, tile_size_y, tile_size_x - (2 * haloZoneSize), haloZoneSize, tile_size_with_halo_x);
  }
  if (!is_left) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize, haloZoneSize * 2, haloZoneSize, tile_size_y - (2 * haloZoneSize), tile_size_with_halo_x);
  }
  if (!is_right) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), tile_size_x, haloZoneSize * 2, haloZoneSize, tile_size_y - (2 * haloZoneSize), tile_size_with_halo_x);
  }

  // small square tiles in the corners
  // in 1d decomposition, all small squares are part of the border which we DO NOT compute
  if (!is_top && !is_left && !decomp_1d) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize, haloZoneSize, haloZoneSize, haloZoneSize, tile_size_with_halo_x);
  }
  if (!is_top && !is_right && !decomp_1d) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), tile_size_x, haloZoneSize, haloZoneSize, haloZoneSize, tile_size_with_halo_x);
  }
  if (!is_bottom && !is_left && !decomp_1d) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize, tile_size_y, haloZoneSize, haloZoneSize, tile_size_with_halo_x);
  }
  if (!is_bottom && !is_right && !decomp_1d) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), tile_size_x, tile_size_y, haloZoneSize, haloZoneSize, tile_size_with_halo_x);
  }
}


// local data should have size (tile_size_with_halo_x * tile_size_with_halo_y)
void ParallelHeatSolver::startHaloExchangeP2P(float* localData, std::array<MPI_Request, 8>& requests) {
/**********************************************************************************************************************/
/*                       Start the non-blocking halo zones exchange using P2P communication.                          */
/*                         Use the requests array to return the requests from the function.                           */
/*                            Don't forget to set the empty requests to MPI_REQUEST_NULL.                             */
/**********************************************************************************************************************/
  
  // left col float
  MPI_Isend(localData, 1, halo_col_left_type_float, neighbors[ND::W], TO_W, cart_comm, &requests[0]);
  MPI_Irecv(localData, 1, halo_col_right_type_float, neighbors[ND::E], FROM_E, cart_comm, &requests[4]);

  // right col float
  MPI_Isend(localData, 1, halo_col_right_type_float, neighbors[ND::E], TO_E, cart_comm, &requests[1]);
  MPI_Irecv(localData, 1, halo_col_left_type_float, neighbors[ND::W], FROM_W, cart_comm, &requests[5]);

  // up row float
  MPI_Isend(localData, 1, halo_row_up_type_float, neighbors[ND::N], TO_N, cart_comm, &requests[2]);
  MPI_Irecv(localData, 1, halo_row_down_type_float, neighbors[ND::S], FROM_S, cart_comm, &requests[6]);

  // down row float
  MPI_Isend(localData, 1, halo_row_down_type_float, neighbors[ND::S], TO_S, cart_comm, &requests[3]);
  MPI_Irecv(localData, 1, halo_row_up_type_float, neighbors[ND::N], FROM_N, cart_comm, &requests[7]);
}

void ParallelHeatSolver::startHaloExchangeRMA(float* localData, MPI_Win window) {
/**********************************************************************************************************************/
/*                       Start the non-blocking halo zones exchange using RMA communication.                          */
/*                   Do not forget that you put/get the values to/from the target's opposite side                     */
/**********************************************************************************************************************/
  // TODO: RMA
}

void ParallelHeatSolver::awaitHaloExchangeP2P(std::array<MPI_Request, 8>& requests) {
/**********************************************************************************************************************/
/*                       Wait for all halo zone exchanges to finalize using P2P communication.                        */
/**********************************************************************************************************************/
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void ParallelHeatSolver::awaitHaloExchangeRMA(MPI_Win window) {
/**********************************************************************************************************************/
/*                       Wait for all halo zone exchanges to finalize using RMA communication.                        */
/**********************************************************************************************************************/


}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>>& outResult) {
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
