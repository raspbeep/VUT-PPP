#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <ios>
#include <string_view>

#include "ParallelHeatSolver.hpp"

ParallelHeatSolver::ParallelHeatSolver(const SimulationProperties& simulationProps,
                                       const MaterialProperties&   materialProps)
: HeatSolverBase(simulationProps, materialProps)
{
  MPI_Comm_size(MPI_COMM_WORLD, &mWorldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mWorldRank);

/**********************************************************************************************************************/
/*                                  Call init* and alloc* methods in correct order                                    */
/**********************************************************************************************************************/
  initGridTopology();
  initDataDistribution();
  allocLocalTiles();
  initHaloExchange();

  if(!mSimulationProps.getOutputFileName().empty())
  {
/**********************************************************************************************************************/
/*                               Open output file if output file name was specified.                                  */
/*  If mSimulationProps.useParallelIO() flag is set to true, open output file for parallel access, otherwise open it  */
/*                         only on MASTER rank using sequetial IO. Use openOutputFile* methods.                       */
/**********************************************************************************************************************/
    if (mSimulationProps.useParallelIO()) {
      // open for parallel access
      openOutputFileParallel();
    } else {
      // open only on root rank
      if (mGridTopologyRank == MPI_ROOT_RANK) {
        openOutputFileSequential();
      }
    }
  }
}

ParallelHeatSolver::~ParallelHeatSolver()
{
/**********************************************************************************************************************/
/*                                  Call deinit* and dealloc* methods in correct order                                */
/*                                             (should be in reverse order)                                           */
/**********************************************************************************************************************/
  deinitHaloExchange();
  deallocLocalTiles();
  deinitDataDistribution();
  deinitGridTopology();
}

std::string_view ParallelHeatSolver::getCodeType() const
{
  return codeType;
}

void ParallelHeatSolver::initGridTopology()
{
/**********************************************************************************************************************/
/*                          Initialize 2D grid topology using non-periodic MPI Cartesian topology.                    */
/*                       Also create a communicator for middle column average temperature computation.                */
/**********************************************************************************************************************/
  // get the decomposition
  mSimulationProps.getDecompGrid(nx, ny);
  int dims[2];
  if (mSimulationProps.getDecomposition() == SimulationProperties::Decomposition::d1) {
    // 1D decomposition
    ndims = 1;
    dims[0] = nx;
    dims[1] = ny;
  } else {
    // 2D decomposition
    ndims = 2;
    dims[0] = ny;
    dims[1] = nx;
  }
  int periods[2] = {false, false};
  int reorder = true;

  // create cartesian topology
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &MPI_GRID_TOPOLOGY);

  // ensure that all ranks are in the grid topology
  if (MPI_GRID_TOPOLOGY == MPI_COMM_NULL) {
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }

  // get rank id and size of the MPI_GRID_TOPOLOGY communicator
  MPI_Comm_rank(MPI_GRID_TOPOLOGY, &mGridTopologyRank);
  MPI_Comm_size(MPI_GRID_TOPOLOGY, &mGridTopologySize);

  // get the position of a rank in the new communicator
  int coord[ndims];
  MPI_Cart_coords(MPI_GRID_TOPOLOGY, mGridTopologyRank, 2, coord);

  // find the neighbours with which the rank will communicate
  if (ndims == 1) {
    // shift by one
    MPI_Cart_shift(MPI_GRID_TOPOLOGY, 0, 1, &neighbours[LEFT], &neighbours[RIGHT]);
    neighbours[TOP] = MPI_PROC_NULL;
    neighbours[BOTTOM] = MPI_PROC_NULL;
    // shift by two
    MPI_Cart_shift(MPI_GRID_TOPOLOGY, 0, 2, &neighboursShiftedByTwo[LEFT], &neighboursShiftedByTwo[RIGHT]);
    neighboursShiftedByTwo[TOP] = MPI_PROC_NULL;
    neighboursShiftedByTwo[BOTTOM] = MPI_PROC_NULL;
  } else {
    // shift by one
    MPI_Cart_shift(MPI_GRID_TOPOLOGY, 1, 1, &neighbours[LEFT], &neighbours[RIGHT]);
    MPI_Cart_shift(MPI_GRID_TOPOLOGY, 0, 1, &neighbours[TOP], &neighbours[BOTTOM]);
    // shift by two
    MPI_Cart_shift(MPI_GRID_TOPOLOGY, 1, 2, &neighboursShiftedByTwo[LEFT], &neighboursShiftedByTwo[RIGHT]);
    MPI_Cart_shift(MPI_GRID_TOPOLOGY, 0, 2, &neighboursShiftedByTwo[TOP], &neighboursShiftedByTwo[BOTTOM]);
  }

  // create a communicator for middle column average temperature computation
  // add ranks in the middle (right side)
  if (ndims == 1) {
    // split by coord[0]
    if (coord[0] == nx/2) { // middle
      MPI_Comm_split(MPI_COMM_WORLD, 0, coord[0], &MPI_MIDDLE_COLUMN_AVG_COMM);
    } else {  // other ranks
      MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, mGridTopologyRank, &MPI_MIDDLE_COLUMN_AVG_COMM);
    }
  } else {
    // split by coord[1]
    if (coord[1] == nx/2) { // middle
      MPI_Comm_split(MPI_COMM_WORLD, 0, coord[0], &MPI_MIDDLE_COLUMN_AVG_COMM);
    } else {  // other ranks
      MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, mGridTopologyRank, &MPI_MIDDLE_COLUMN_AVG_COMM);
    }
  }
  
  // get rank id and size of the MPI_MIDDLE_COLUMN_AVG_COMM communicator
  if (MPI_MIDDLE_COLUMN_AVG_COMM != MPI_COMM_NULL) {
    MPI_Comm_rank(MPI_MIDDLE_COLUMN_AVG_COMM, &mMiddleColumnAvgRank);
    MPI_Comm_size(MPI_MIDDLE_COLUMN_AVG_COMM, &mMiddleColumnAvgSize);
  } else {  // other ranks
    mMiddleColumnAvgRank = MPI_PROC_NULL;
    mMiddleColumnAvgSize = 0;
  }
}

void ParallelHeatSolver::deinitGridTopology()
{
/**********************************************************************************************************************/
/*      Deinitialize 2D grid topology and the middle column average temperature computation communicator              */
/**********************************************************************************************************************/
  // free the MPI_GRID_TOPOLOGY communicator
  MPI_Comm_free(&MPI_GRID_TOPOLOGY);

  // free the MPI_MIDDLE_COLUMN_AVG_COMM communicator
  if (MPI_MIDDLE_COLUMN_AVG_COMM != MPI_COMM_NULL) {
    MPI_Comm_free(&MPI_MIDDLE_COLUMN_AVG_COMM);
  }
}

void ParallelHeatSolver::initDataDistribution()
{
/**********************************************************************************************************************/
/*                 Initialize variables and MPI datatypes for data distribution (float and int).                      */
/**********************************************************************************************************************/
  tileWidth = mMaterialProps.getEdgeSize() / nx;
  tileHeight = mMaterialProps.getEdgeSize() / ny;
  width = 2 * haloZoneSize + tileWidth;

  // create local tile types
  int size[2];
  size[0] = 2 * haloZoneSize + tileHeight;
  size[1] = 2 * haloZoneSize + tileWidth;
  int tile[2] = {int(tileHeight), int(tileWidth)};
  int start[2] = {int(haloZoneSize), int(haloZoneSize)};
  // create subarray type for domain map (int)
  MPI_Type_create_subarray(2, size, tile, start, MPI_ORDER_C, MPI_INT, &MPI_WORKER_TYPE_INT);
  MPI_Type_commit(&MPI_WORKER_TYPE_INT);
  // create subarray type for domain parameters and temperature (float)
  MPI_Type_create_subarray(2, size, tile, start, MPI_ORDER_C, MPI_FLOAT, &MPI_WORKER_TYPE_FLOAT);
  MPI_Type_commit(&MPI_WORKER_TYPE_FLOAT);
  // save tile sizes
  globalDataSize = mMaterialProps.getEdgeSize() * mMaterialProps.getEdgeSize();
  localDataSize = size[0] * size[1];
  
  // create global tile types
  size[0] = mMaterialProps.getEdgeSize();
  size[1] = mMaterialProps.getEdgeSize();
  start[0] = 0;
  start[1] = 0;
  // create subarray type for global domain map (int)
  MPI_Type_create_subarray(2, size, tile, start, MPI_ORDER_C, MPI_INT, &MPI_FARMER_TYPE_INT);
  MPI_Type_commit(&MPI_FARMER_TYPE_INT);
  // resized data type
  MPI_Type_create_resized(MPI_FARMER_TYPE_INT, 0, 1*sizeof(int), &MPI_FARMER_TYPE_RESIZED_INT);
  MPI_Type_commit(&MPI_FARMER_TYPE_RESIZED_INT);
  // create subarray type for global domain parameters and temperature (float)
  MPI_Type_create_subarray(2, size, tile, start, MPI_ORDER_C, MPI_FLOAT, &MPI_FARMER_TYPE_FLOAT);
  MPI_Type_commit(&MPI_FARMER_TYPE_FLOAT);
  // resized data type
  MPI_Type_create_resized(MPI_FARMER_TYPE_FLOAT, 0, 1*sizeof(float), &MPI_FARMER_TYPE_RESIZED_FLOAT);
  MPI_Type_commit(&MPI_FARMER_TYPE_RESIZED_FLOAT);  
}

void ParallelHeatSolver::deinitDataDistribution()
{
/**********************************************************************************************************************/
/*                       Deinitialize variables and MPI datatypes for data distribution.                              */
/**********************************************************************************************************************/
  MPI_Type_free(&MPI_WORKER_TYPE_INT);
  MPI_Type_free(&MPI_WORKER_TYPE_FLOAT);

  MPI_Type_free(&MPI_FARMER_TYPE_INT);
  MPI_Type_free(&MPI_FARMER_TYPE_RESIZED_INT);
  MPI_Type_free(&MPI_FARMER_TYPE_FLOAT);
  MPI_Type_free(&MPI_FARMER_TYPE_RESIZED_FLOAT);
}

void ParallelHeatSolver::allocLocalTiles()
{
/**********************************************************************************************************************/
/*            Allocate local tiles for domain map (1x), domain parameters (1x) and domain temperature (2x).           */
/*                                               Use AlignedAllocator.                                                */
/**********************************************************************************************************************/
  domainMapLocal.resize(localDataSize);
  domainParametersLocal.resize(localDataSize);
  tempArraysLocal[0].resize(localDataSize);
  tempArraysLocal[1].resize(localDataSize);
}

void ParallelHeatSolver::deallocLocalTiles()
{
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
  // with top and bottom row the MPI_HALO_ROW_TYPE is exchanged
  MPI_Type_vector(haloZoneSize, tileWidth, width, MPI_FLOAT, &MPI_HALO_ROW_TYPE_FLOAT);
  MPI_Type_commit(&MPI_HALO_ROW_TYPE_FLOAT);

  // with left and right neighbour the MPI_HALO_COL_TYPE is exchanged
  MPI_Type_vector(tileHeight, haloZoneSize, width, MPI_FLOAT, &MPI_HALO_COL_TYPE_FLOAT);
  MPI_Type_commit(&MPI_HALO_COL_TYPE_FLOAT);

  // create RMA windows
  if (mSimulationProps.isRunParallelRMA()) {
    // win 0
    MPI_Win_create(tempArraysLocal[0].data(), localDataSize * sizeof(float), sizeof(float), 
                   MPI_INFO_NULL, MPI_GRID_TOPOLOGY, &MPI_HALO_TILE_WIN[0]);
    MPI_Win_fence(0, MPI_HALO_TILE_WIN[0]);

    // win 1
    MPI_Win_create(tempArraysLocal[1].data(), localDataSize * sizeof(float), sizeof(float), 
                   MPI_INFO_NULL, MPI_GRID_TOPOLOGY, &MPI_HALO_TILE_WIN[1]);
    MPI_Win_fence(0, MPI_HALO_TILE_WIN[1]);
  }
}

void ParallelHeatSolver::deinitHaloExchange()
{
/**********************************************************************************************************************/
/*                            Deinitialize variables and MPI datatypes for halo exchange.                             */
/**********************************************************************************************************************/
  MPI_Type_free(&MPI_HALO_ROW_TYPE_FLOAT);
  MPI_Type_free(&MPI_HALO_COL_TYPE_FLOAT);

  if (mSimulationProps.isRunParallelRMA()) {
    MPI_Win_free(&MPI_HALO_TILE_WIN[0]);
    MPI_Win_free(&MPI_HALO_TILE_WIN[1]);
  }
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
  // compute counts and displacements
  std::vector<int> counts{};
  std::vector<int> displacements{};
  for (int rank = 0; rank < mGridTopologySize; rank++) {
    counts.push_back(1);
    displacements.push_back((rank / nx) * mMaterialProps.getEdgeSize() * tileHeight + (rank % nx) * tileWidth);
  }
  // choose the data type
  MPI_Datatype GLOBAL_TILE_TYPE = std::is_same_v<T, int> ? MPI_FARMER_TYPE_RESIZED_INT : MPI_FARMER_TYPE_RESIZED_FLOAT;
  MPI_Datatype LOCAL_TILE_TYPE = std::is_same_v<T, int> ? MPI_WORKER_TYPE_INT : MPI_WORKER_TYPE_FLOAT;

  // scatter the array
  MPI_Scatterv(globalData, counts.data(), displacements.data(), GLOBAL_TILE_TYPE, 
               localData, 1, LOCAL_TILE_TYPE, MPI_ROOT_RANK, MPI_GRID_TOPOLOGY);
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
  // compute counts and displacements
  std::vector<int> counts{};
  std::vector<int> displacements{};
  for (int rank = 0; rank < mGridTopologySize; rank++) {
    counts.push_back(1);
    displacements.push_back((rank / nx) * mMaterialProps.getEdgeSize() * tileHeight + (rank % nx) * tileWidth);
  }
  // choose the data type
  MPI_Datatype GLOBAL_TILE_TYPE = std::is_same_v<T, int> ? MPI_FARMER_TYPE_RESIZED_INT : MPI_FARMER_TYPE_RESIZED_FLOAT;
  MPI_Datatype LOCAL_TILE_TYPE = std::is_same_v<T, int> ? MPI_WORKER_TYPE_INT : MPI_WORKER_TYPE_FLOAT;

  // gather the array
  MPI_Gatherv(localData, 1, LOCAL_TILE_TYPE,
              globalData, counts.data(), displacements.data(), GLOBAL_TILE_TYPE,
              MPI_ROOT_RANK, MPI_GRID_TOPOLOGY);
}

void ParallelHeatSolver::computeHaloZones(const float* oldTemp, float* newTemp)
{
/**********************************************************************************************************************/
/*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
/*                        Use updateTile method to compute new temperatures in halo zones.                            */
/*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
/**********************************************************************************************************************/
  // update edges
  // update top halo zone
  if (neighbours[TOP] != MPI_PROC_NULL) {
    updateTile(oldTemp, newTemp, domainParametersLocal.data(), domainMapLocal.data(), 
              2*haloZoneSize, haloZoneSize, 
              tileWidth - 2*haloZoneSize, haloZoneSize, 
              width);
  }
  
  // update bottom halo zone
  if (neighbours[BOTTOM] != MPI_PROC_NULL && tileHeight > haloZoneSize) {
    updateTile(oldTemp, newTemp, domainParametersLocal.data(), domainMapLocal.data(), 
              2*haloZoneSize, tileHeight, 
              tileWidth - 2*haloZoneSize, haloZoneSize, 
              width);
  }

  // update left halo zone
  if (neighbours[LEFT] != MPI_PROC_NULL) {
    // compute only part of left 
    updateTile(oldTemp, newTemp, domainParametersLocal.data(), domainMapLocal.data(), 
              haloZoneSize, 2 * haloZoneSize, 
              haloZoneSize, tileHeight - 2*haloZoneSize, 
              width);
  }

  // update right halo zone
  if (neighbours[RIGHT] != MPI_PROC_NULL) {
    updateTile(oldTemp, newTemp, domainParametersLocal.data(), domainMapLocal.data(), 
              tileWidth, 2 * haloZoneSize, 
              haloZoneSize, tileHeight - 2*haloZoneSize, 
              width);
  }

  // update corners
  if (neighbours[LEFT] != MPI_PROC_NULL && neighbours[TOP] != MPI_PROC_NULL) {
    // update left top corner
    updateTile(oldTemp, newTemp, domainParametersLocal.data(), domainMapLocal.data(), 
              haloZoneSize, haloZoneSize, 
              haloZoneSize, haloZoneSize, 
              width);
  }
  if (neighbours[TOP] != MPI_PROC_NULL && neighbours[RIGHT] != MPI_PROC_NULL) {
    // update right top corner
    updateTile(oldTemp, newTemp, domainParametersLocal.data(), domainMapLocal.data(), 
              tileWidth, haloZoneSize, 
              haloZoneSize, haloZoneSize, 
              width);
  }
  if (neighbours[RIGHT] != MPI_PROC_NULL && neighbours[BOTTOM] != MPI_PROC_NULL) {
    // update right bottom corner
    updateTile(oldTemp, newTemp, domainParametersLocal.data(), domainMapLocal.data(), 
              tileWidth, tileHeight, 
              haloZoneSize, haloZoneSize, 
              width);
  }
  if (neighbours[BOTTOM] != MPI_PROC_NULL && neighbours[LEFT] != MPI_PROC_NULL) {
    // update left bottom corner
    updateTile(oldTemp, newTemp, domainParametersLocal.data(), domainMapLocal.data(), 
              haloZoneSize, tileHeight, 
              haloZoneSize, haloZoneSize, 
              width);
  }
}

void ParallelHeatSolver::startHaloExchangeP2P(float* localData, std::array<MPI_Request, 8>& requests)
{
/**********************************************************************************************************************/
/*                       Start the non-blocking halo zones exchange using P2P communication.                          */
/*                         Use the requests array to return the requests from the function.                           */
/*                            Don't forget to set the empty requests to MPI_REQUEST_NULL.                             */
/**********************************************************************************************************************/
  // send
  // left neighbour
  if (neighbours[LEFT] == MPI_PROC_NULL 
                                || (neighboursShiftedByTwo[LEFT] == MPI_PROC_NULL && (tileWidth <= haloZoneSize))) {
    // do not wait
    requests[LEFT] = MPI_REQUEST_NULL;
  } else {
    MPI_Isend(localData + haloZoneSize * width + haloZoneSize, 1, MPI_HALO_COL_TYPE_FLOAT, 
              neighbours[LEFT], 0, MPI_GRID_TOPOLOGY, &requests[LEFT]);
  }
  // top neighbour
  if (neighbours[TOP] == MPI_PROC_NULL 
                                || (neighboursShiftedByTwo[TOP] == MPI_PROC_NULL && tileHeight <= haloZoneSize)) {
    // do not wait
    requests[TOP] = MPI_REQUEST_NULL;
  } else {
    MPI_Isend(localData + haloZoneSize * width + haloZoneSize, 1, MPI_HALO_ROW_TYPE_FLOAT, 
              neighbours[TOP], 0, MPI_GRID_TOPOLOGY, &requests[TOP]);
  }
  // right neighbour
  if (neighbours[RIGHT] == MPI_PROC_NULL 
                                || (neighboursShiftedByTwo[RIGHT] == MPI_PROC_NULL && (tileWidth <= haloZoneSize))) {
    // do not wait
    requests[RIGHT] = MPI_REQUEST_NULL;
  } else {
    MPI_Isend(localData + haloZoneSize * width + tileWidth, 1, MPI_HALO_COL_TYPE_FLOAT, 
              neighbours[RIGHT], 0, MPI_GRID_TOPOLOGY, &requests[RIGHT]);
  }
  // bottom neighbour
  if (neighbours[BOTTOM] == MPI_PROC_NULL 
                                || (neighboursShiftedByTwo[BOTTOM] == MPI_PROC_NULL && tileHeight <= haloZoneSize)) {
    // do not wait
    requests[BOTTOM] = MPI_REQUEST_NULL;
  } else {
    MPI_Isend(localData + width * tileHeight + haloZoneSize, 1, MPI_HALO_ROW_TYPE_FLOAT, 
              neighbours[BOTTOM], 0, MPI_GRID_TOPOLOGY, &requests[BOTTOM]);
  }

  // receive
  // left neighbour 
  if (neighbours[LEFT] == MPI_PROC_NULL || (neighbours[RIGHT] == MPI_PROC_NULL && tileWidth <= haloZoneSize)) {
    requests[4] = MPI_REQUEST_NULL;
  } else {
    MPI_Irecv(localData + haloZoneSize * width, 
              1, MPI_HALO_COL_TYPE_FLOAT, neighbours[LEFT], MPI_ANY_TAG, MPI_GRID_TOPOLOGY, &requests[4]);
  }
  // top neighbour 
  if (neighbours[TOP] == MPI_PROC_NULL || (neighbours[BOTTOM] == MPI_PROC_NULL && tileWidth <= haloZoneSize)) {
    // do not wait
    requests[5] = MPI_REQUEST_NULL;
  } else {
    MPI_Irecv(localData + haloZoneSize, 
              1, MPI_HALO_ROW_TYPE_FLOAT, neighbours[TOP], MPI_ANY_TAG, MPI_GRID_TOPOLOGY, &requests[5]);
  }
  // right neighbour 
  if (neighbours[RIGHT] == MPI_PROC_NULL || (neighbours[LEFT] == MPI_PROC_NULL && tileWidth <= haloZoneSize)) {
    // do not wait
    requests[6] = MPI_REQUEST_NULL;
  } else {
    MPI_Irecv(localData + haloZoneSize * width + tileWidth + haloZoneSize, 
              1, MPI_HALO_COL_TYPE_FLOAT, neighbours[RIGHT], MPI_ANY_TAG, MPI_GRID_TOPOLOGY, &requests[6]);
  }
  // bottom neighbour 
  if (neighbours[BOTTOM] == MPI_PROC_NULL || (neighbours[TOP] == MPI_PROC_NULL && tileWidth <= haloZoneSize)) {
    // do not wait
    requests[7] = MPI_REQUEST_NULL;
  } else {
    MPI_Irecv(localData + (haloZoneSize + tileHeight) * width + haloZoneSize, 
              1, MPI_HALO_ROW_TYPE_FLOAT, neighbours[BOTTOM], MPI_ANY_TAG, MPI_GRID_TOPOLOGY, &requests[7]);
  }
}

void ParallelHeatSolver::startHaloExchangeRMA(float* localData, MPI_Win window)
{
/**********************************************************************************************************************/
/*                       Start the non-blocking halo zones exchange using RMA communication.                          */
/*                   Do not forget that you put/get the values to/from the target's opposite side                     */
/**********************************************************************************************************************/
  MPI_Win_fence(0, window); // open the window

  // left neighbour
  if (neighbours[LEFT] != MPI_PROC_NULL 
                              && !(neighboursShiftedByTwo[LEFT] == MPI_PROC_NULL && (tileWidth <= haloZoneSize))) {
    MPI_Put(localData + haloZoneSize * width + haloZoneSize, 1, MPI_HALO_COL_TYPE_FLOAT, 
            neighbours[LEFT], haloZoneSize * width + tileWidth + haloZoneSize, 1, MPI_HALO_COL_TYPE_FLOAT, window);
  }
  // top neighbour
  if (neighbours[TOP] != MPI_PROC_NULL 
                              && !(neighboursShiftedByTwo[TOP] == MPI_PROC_NULL && (tileHeight <= haloZoneSize))) {
    MPI_Put(localData + haloZoneSize * width + haloZoneSize, 1, MPI_HALO_ROW_TYPE_FLOAT, 
            neighbours[TOP], width * (tileHeight + haloZoneSize) + haloZoneSize, 1, MPI_HALO_ROW_TYPE_FLOAT, window);
  }
  // right neighbour
  if (neighbours[RIGHT] != MPI_PROC_NULL 
                              && !(neighboursShiftedByTwo[RIGHT] == MPI_PROC_NULL && (tileWidth <= haloZoneSize))) {
    MPI_Put(localData + haloZoneSize * width + tileWidth, 1, MPI_HALO_COL_TYPE_FLOAT, 
            neighbours[RIGHT], haloZoneSize * width, 1, MPI_HALO_COL_TYPE_FLOAT, window);
  }
  // bottom neighbour
  if (neighbours[BOTTOM] != MPI_PROC_NULL 
                              && !(neighboursShiftedByTwo[BOTTOM] == MPI_PROC_NULL && (tileHeight <= haloZoneSize))) {
    MPI_Put(localData + width * tileHeight + haloZoneSize, 1, MPI_HALO_ROW_TYPE_FLOAT, 
            neighbours[BOTTOM], haloZoneSize, 1, MPI_HALO_ROW_TYPE_FLOAT, window);
  }
}

void ParallelHeatSolver::awaitHaloExchangeP2P(std::array<MPI_Request, 8>& requests)
{
/**********************************************************************************************************************/
/*                       Wait for all halo zone exchanges to finalize using P2P communication.                        */
/**********************************************************************************************************************/
  MPI_Waitall(8, requests.data(), MPI_STATUS_IGNORE);
}

void ParallelHeatSolver::awaitHaloExchangeRMA(MPI_Win window)
{
/**********************************************************************************************************************/
/*                       Wait for all halo zone exchanges to finalize using RMA communication.                        */
/**********************************************************************************************************************/
  MPI_Win_fence(0, window); // close the window
}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>>& outResult)
{
  std::array<MPI_Request, 8> requestsP2P{};

/**********************************************************************************************************************/
/*                                         Scatter initial data.                                                      */
/**********************************************************************************************************************/
  scatterTiles(mMaterialProps.getDomainMap().data(), domainMapLocal.data());
  scatterTiles(mMaterialProps.getDomainParameters().data(), domainParametersLocal.data());
  scatterTiles(mMaterialProps.getInitialTemperature().data(), tempArraysLocal[0].data());

/**********************************************************************************************************************/
/* Exchange halo zones of initial domain temperature and parameters using P2P communication. Wait for them to finish. */
/**********************************************************************************************************************/
  // start sending initial temperature
  std::array <MPI_Request, 8> tempRequests;
  startHaloExchangeP2P(tempArraysLocal[0].data(), tempRequests);

  // start sending domain parameters
  std::array <MPI_Request, 8> paramsRequests;
  startHaloExchangeP2P(domainParametersLocal.data(), paramsRequests);

  // wait for initial temperature
  awaitHaloExchangeP2P(tempRequests);

  // wait for domain parameters 
  awaitHaloExchangeP2P(paramsRequests);

/**********************************************************************************************************************/
/*                            Copy initial temperature to the second buffer.                                          */
/**********************************************************************************************************************/
  tempArraysLocal[1] = tempArraysLocal[0];

  double startTime = MPI_Wtime();

  // 3. Start main iterative simulation loop.
  for(std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter)
  {
    const std::size_t oldIdx = iter % 2;       // Index of the buffer with old temperatures
    const std::size_t newIdx = (iter + 1) % 2; // Index of the buffer with new temperatures

/**********************************************************************************************************************/
/*                            Compute and exchange halo zones using P2P or RMA.                                       */
/**********************************************************************************************************************/
    computeHaloZones(tempArraysLocal[oldIdx].data(), tempArraysLocal[newIdx].data());
    
    if (mSimulationProps.isRunParallelP2P()) {
      startHaloExchangeP2P(tempArraysLocal[newIdx].data(), requestsP2P);
    } else if (mSimulationProps.isRunParallelRMA()) {
      startHaloExchangeRMA(tempArraysLocal[newIdx].data(), MPI_HALO_TILE_WIN[newIdx]);
    }
/**********************************************************************************************************************/
/*                           Compute the rest of the tile. Use updateTile method.                                     */
/**********************************************************************************************************************/
    updateTile(tempArraysLocal[oldIdx].data(), tempArraysLocal[newIdx].data(), 
              domainParametersLocal.data(), domainMapLocal.data(), 
              2 * haloZoneSize, 2 * haloZoneSize, 
              tileWidth - 2 * haloZoneSize, tileHeight - 2 * haloZoneSize, 
              2 * haloZoneSize + tileWidth);
/**********************************************************************************************************************/
/*                            Wait for all halo zone exchanges to finalize.                                           */
/**********************************************************************************************************************/
    if (mSimulationProps.isRunParallelP2P()) {
      awaitHaloExchangeP2P(requestsP2P);
    } else if (mSimulationProps.isRunParallelRMA()) {
      awaitHaloExchangeRMA(MPI_HALO_TILE_WIN[newIdx]);
    }

    if(shouldStoreData(iter))
    {
/**********************************************************************************************************************/
/*                          Store the data into the output file using parallel or sequential IO.                      */
/**********************************************************************************************************************/
      if (mSimulationProps.useParallelIO()) {
        storeDataIntoFileParallel(mFileHandle, iter, tempArraysLocal[newIdx].data());
      } else {
        // gather data
        gatherTiles(tempArraysLocal[newIdx].data(), outResult.data());
        // store data into file by root rank
        storeDataIntoFileSequential(mFileHandle, iter, outResult.data());
      }
    }

    if(shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature())
    {
/**********************************************************************************************************************/
/*                 Compute and print middle column average temperature and print progress report.                     */
/**********************************************************************************************************************/
      float middleColAvgTemp = computeMiddleColumnAverageTemperatureParallel(tempArraysLocal[newIdx].data());
      if (mMiddleColumnAvgRank == MPI_ROOT_RANK) {
        printProgressReport(iter, middleColAvgTemp);
      }
    }
  }

  const std::size_t resIdx = mSimulationProps.getNumIterations() % 2; // Index of the buffer with final temperatures

  double elapsedTime = MPI_Wtime() - startTime;

/**********************************************************************************************************************/
/*                                     Gather final domain temperature.                                               */
/**********************************************************************************************************************/
  gatherTiles(tempArraysLocal[resIdx].data(), outResult.data());

/**********************************************************************************************************************/
/*           Compute (sequentially) and report final middle column temperature average and print final report.        */
/**********************************************************************************************************************/
  if (mGridTopologyRank == MPI_ROOT_RANK) {
    float middleColAvgTemp = computeMiddleColumnAverageTemperatureSequential(outResult.data());
    printFinalReport(elapsedTime, middleColAvgTemp);
  }

}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const
{
/**********************************************************************************************************************/
/*                Return true if rank should compute middle column average temperature.                               */
/**********************************************************************************************************************/
  if (MPI_MIDDLE_COLUMN_AVG_COMM != MPI_COMM_NULL) {
    return true;
  }

  return false;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureParallel(const float *localData) const
{
/**********************************************************************************************************************/
/*                  Implement parallel middle column average temperature computation.                                 */
/*                      Use OpenMP directives to accelerate the local computations.                                   */
  float middleColAvgTemp{};
  float localSum{};
  float globalSum{};

  size_t tileEnd = localDataSize - haloZoneSize * width;
  size_t startingOffset{};

  if (ndims == 2 && nx == 1) {
    // compute local sum in the middle column
    startingOffset = haloZoneSize*width + (width / 2);
  } else {
    // compute local sum in the most left column
    startingOffset = haloZoneSize + haloZoneSize*width;
  }

  // compute local sum
  #pragma omp parallel for reduction(+:localSum)
  for (size_t i = startingOffset; i < tileEnd; i=i+width) {
    localSum += localData[i];
  }

  // get the global sum (by root)
  MPI_Reduce(&localSum, &globalSum, 1, MPI_FLOAT, MPI_SUM, MPI_ROOT_RANK, MPI_MIDDLE_COLUMN_AVG_COMM);

  // compute the average (by root)
  if (mMiddleColumnAvgRank == MPI_ROOT_RANK) {
    middleColAvgTemp = globalSum / static_cast<float>(tileHeight * mMiddleColumnAvgSize);
  }

  return middleColAvgTemp;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureSequential(const float *globalData) const
{  
/**********************************************************************************************************************/
/*                  Implement sequential middle column average temperature computation.                               */
/*                      Use OpenMP directives to accelerate the local computations.                                   */
/**********************************************************************************************************************/
  float middleColAvgTemp{};
  int numRows = mMaterialProps.getEdgeSize();

  #pragma omp parallel for reduction(+:middleColAvgTemp)
  for(int i = 0; i < numRows; ++i) {
      middleColAvgTemp += globalData[i * numRows + numRows / 2];
  }

  return middleColAvgTemp / static_cast<float>(numRows);
}

void ParallelHeatSolver::openOutputFileSequential()
{
  // create the output file for sequential access
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
  faplHandle = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(faplHandle, MPI_GRID_TOPOLOGY, MPI_INFO_NULL);

  // alignment for file access
  hsize_t alignment = 4096;
  H5Pset_alignment(faplHandle, 0, alignment);

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
    std::array<hsize_t, 2> tileSize{tileHeight, tileWidth};
    std::array<hsize_t, 2> localDataTileOffset{haloZoneSize, haloZoneSize};

    hsize_t offsetX{};
    hsize_t offsetY{};

      // check the decomposition
    if (ndims == 1) {
      // 1D decomposition
      offsetY = 0;
      offsetX = mGridTopologyRank * tileWidth;
    } else {
      // 2D decomposition
      int coord[2];
      MPI_Cart_coords(MPI_GRID_TOPOLOGY, mGridTopologyRank, 2, coord);
      offsetX = coord[1] * tileWidth;
      offsetY = coord[0] * tileHeight;
    }

    std::array<hsize_t, 2> tileOffsetInGlobal{offsetY, offsetX};

    // Create new dataspace and dataset using it.
    static constexpr std::string_view dataSetName{"Temperature"};

    Hdf5PropertyListHandle datasetPropListHandle{};
/**********************************************************************************************************************/
/*                            Create dataset property list to set up chunking.                                        */
/*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
/**********************************************************************************************************************/
    datasetPropListHandle = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(datasetPropListHandle, 2, tileSize.data());

    Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
    Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                              H5T_NATIVE_FLOAT, dataSpaceHandle,
                                              H5P_DEFAULT, datasetPropListHandle,
                                              H5P_DEFAULT));

    Hdf5DataspaceHandle memSpaceHandle{};
/**********************************************************************************************************************/
/*                Create memory dataspace representing tile in the memory (set up memSpaceHandle).                    */
/**********************************************************************************************************************/
    std::array<hsize_t, 2> tileSizeWithHaloZones{tileHeight + 2*haloZoneSize, tileWidth + 2*haloZoneSize};
    memSpaceHandle = H5Screate_simple(2, tileSizeWithHaloZones.data(), nullptr);

/**********************************************************************************************************************/
/*              Select inner part of the tile in memory and matching part of the dataset in the file                  */
/*                           (given by position of the tile in global domain).                                        */
/**********************************************************************************************************************/
    H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, localDataTileOffset.data(), nullptr, tileSize.data(), nullptr);
    H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, tileOffsetInGlobal.data(), nullptr, tileSize.data(), nullptr);

    Hdf5PropertyListHandle propListHandle{};

/**********************************************************************************************************************/
/*              Perform collective write operation, writting tiles from all processes at once.                        */
/*                                   Set up the propListHandle variable.                                              */
/**********************************************************************************************************************/
    // create XFER property list and set collective IO
    propListHandle = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(propListHandle, H5FD_MPIO_COLLECTIVE);
    
    // write
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
