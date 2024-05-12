/**
 * @file    ParallelHeatSolver.cpp
 *
 * @author  Pavel Kratochvíl <xkrato61@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-04-26
 *
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <ios>
#include <string_view>

#include "ParallelHeatSolver.hpp"

ParallelHeatSolver::ParallelHeatSolver(const SimulationProperties &simulationProps,
                                       const MaterialProperties &materialProps)
    : HeatSolverBase(simulationProps, materialProps) {
  MPI_Comm_size(MPI_COMM_WORLD, &mWorldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mWorldRank);

  /**********************************************************************************************************************/
  /*                                  Call init* and alloc* methods in correct order                                    */
  /**********************************************************************************************************************/
  initGridTopology();
  initDataDistribution();
  allocLocalTiles();
  initHaloExchange();

  if (!mSimulationProps.getOutputFileName().empty()) {
    /**********************************************************************************************************************/
    /*                               Open output file if output file name was specified.                                  */
    /*  If mSimulationProps.useParallelIO() flag is set to true, open output file for parallel access, otherwise open it  */
    /*                         only on MASTER rank using sequetial IO. Use openOutputFile* methods.                       */
    /**********************************************************************************************************************/
    if (mSimulationProps.useParallelIO()) {
      openOutputFileParallel();
    } else if (mWorldRank == 0) { 
      openOutputFileSequential();
    }
  }
}

ParallelHeatSolver::~ParallelHeatSolver() {
  /**********************************************************************************************************************/
  /*                                  Call deinit* and dealloc* methods in correct order                                */
  /*                                             (should be in reverse order)                                           */
  /**********************************************************************************************************************/
  deinitDataDistribution();
  deinitHaloExchange();
  deinitGridTopology();
  // not necessary
  // deallocLocalTiles();
}

std::string_view ParallelHeatSolver::getCodeType() const {
  return codeType;
}

void ParallelHeatSolver::initGridTopology() {
  /**********************************************************************************************************************/
  /*                          Initialize 2D grid topology using non-periodic MPI Cartesian topology.                    */
  /*                       Also create a communicator for middle column average temperature computation.                */
  /**********************************************************************************************************************/
  mSimulationProps.getDecompGrid(nx, ny);
  std::array periods = {0, 0};
  decomp = mSimulationProps.getDecomposition() == SimulationProperties::Decomposition::d1 ? 1 : 2;
  
  // get the correct dimensions for cartesian topology
  if (decomp == 1) {
    n_dims = 1;
    dims[0] = nx;
    dims[1] = 1;
  } else {
    n_dims = 2;
    dims[0] = ny;
    dims[1] = nx;
  }
  
  MPI_Bcast(&n_dims, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Cart_create(MPI_COMM_WORLD, n_dims, dims.data(), periods.data(), 1, &cart_comm);
  
  // get rank of in cartesian communicator and the coordinates of individual processors
  MPI_Comm_rank(cart_comm, &mCartRank);
  MPI_Cart_coords(cart_comm, mCartRank, n_dims, cart_coords.data());

  // optimisation for only calculating this once
  // rank's tile position for computeHaloZones()
  if (decomp == 1) {
    is_top = true;
    is_bottom = true;
    is_left = cart_coords[0] == 0;
    is_right = cart_coords[0] == dims[0] - 1;
  } else {
    is_top = cart_coords[0] == 0;
    is_bottom = cart_coords[0] == dims[0] - 1;
    is_left = cart_coords[1] == 0;
    is_right = cart_coords[1] == dims[1] - 1;
  }

  std::fill_n(neighbors.data(), 4, MPI_PROC_NULL);

  // calculate neighbors, MPI automatically assigns MPI_PROC_NULL
  if (n_dims == 2) {
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[0], &neighbors[2]);
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[1], &neighbors[3]);
  } else {
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[0], &neighbors[2]);
  }

  // create the center column communicator for average temperature computation
  should_compute_average = shouldComputeMiddleColumnAverageTemperature();
  if (should_compute_average) {
    MPI_Comm_split(MPI_COMM_WORLD, 0, mWorldRank, &center_col_comm);
  } else {
    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, mWorldRank, &center_col_comm);
  }

  // get the center col rank for progress reporting
  if (center_col_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(center_col_comm, &mCenterColCommRank);
  }
}

void ParallelHeatSolver::deinitGridTopology() {
  /**********************************************************************************************************************/
  /*      Deinitialize 2D grid topology and the middle column average temperature computation communicator              */
  /**********************************************************************************************************************/
  // all ranks are in cartesian grid communicator
  MPI_Comm_free(&cart_comm);
  // only center col ranks free the communicator
  if (center_col_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&center_col_comm);
  }
}

void ParallelHeatSolver::initDataDistribution() {
  /**********************************************************************************************************************/
  /*                 Initialize variables and MPI datatypes for data distribution (float and int).                      */
  /**********************************************************************************************************************/
  // global tile size
  global_edge_size = mMaterialProps.getEdgeSize();
  // calculate the local tile size
  if (decomp == 2) {
    tile_size_x = global_edge_size / dims[1];
    tile_size_y = global_edge_size / dims[0];
  } else {
    tile_size_x = global_edge_size / dims[0];
    tile_size_y = global_edge_size / dims[1];
  }

  // size of the local array with halo border around it
  tile_size_with_halo_x = tile_size_x + (2 * haloZoneSize);
  tile_size_with_halo_y = tile_size_y + (2 * haloZoneSize);

  // only root needs the tile type
  if (mWorldRank == 0) {
    std::array domain_dims = {global_edge_size, global_edge_size};
    std::array tile_dims = {tile_size_y, tile_size_x};
    std::array start_arr = {0, 0};

    // initial datatypes to derive the resized types from
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

    // free intermediate obsolete types
    MPI_Type_free(&tile_org_type_float);
    MPI_Type_free(&tile_org_type_int);
  }

  // tile with a border of two around on the edges
  local_tile_with_halo_dims[0] = tile_size_with_halo_y;
  local_tile_with_halo_dims[1] = tile_size_with_halo_x;

  // original tile size for receiving
  local_tile_dims[0] = tile_size_y;
  local_tile_dims[1] = tile_size_x;
  
  // offset of the local subarray
  start_arr[0] = haloZoneSize;
  start_arr[1] = haloZoneSize;

  // create local tile with haloZoneSize offset
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), local_tile_dims.data(), start_arr.data(), MPI_ORDER_C, MPI_FLOAT, &local_tile_type_float);
  MPI_Type_commit(&local_tile_type_float);

  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), local_tile_dims.data(), start_arr.data(), MPI_ORDER_C, MPI_INT, &local_tile_type_int);
  MPI_Type_commit(&local_tile_type_int);

  // calculate the counts and displacements for tile distribution from root rank
  // counts = nx * ny = world_size = n_processors
  counts = std::make_unique<int[]>(mWorldSize);
  displacements = std::make_unique<int[]>(mWorldSize);
  std::fill_n(counts.get(), mWorldSize, 1);

  int nX = global_edge_size / tile_size_x;

  for (int i = 0; i < mWorldSize; i++ ) {
    int row = i / nX;
    int col = i % nX;
    displacements[i] = (row * tile_size_y * global_edge_size) + (col * tile_size_x);
  }
}

void ParallelHeatSolver::deinitDataDistribution() {
  /**********************************************************************************************************************/
  /*                       Deinitialize variables and MPI datatypes for data distribution.                              */
  /**********************************************************************************************************************/
  // global tile type is specific to root
  if (mWorldRank == 0) {
    MPI_Type_free(&global_tile_type_float);
    MPI_Type_free(&global_tile_type_int);
  }
  
  // free local tile types
  MPI_Type_free(&local_tile_type_float);
  MPI_Type_free(&local_tile_type_int);
}

void ParallelHeatSolver::allocLocalTiles() {
  /**********************************************************************************************************************/
  /*            Allocate local tiles for domain map (1x), domain parameters (1x) and domain temperature (2x).           */
  /*                                               Use AlignedAllocator.                                                */
  /**********************************************************************************************************************/
  tile_map.resize(tile_size_with_halo_x * tile_size_with_halo_y);
  tile_params.resize(tile_size_with_halo_x * tile_size_with_halo_y);

  tile_temps[OLD].resize(tile_size_with_halo_x * tile_size_with_halo_y);
  tile_temps[NEW].resize(tile_size_with_halo_x * tile_size_with_halo_y);
}

void ParallelHeatSolver::deallocLocalTiles() {
  /**********************************************************************************************************************/
  /*                                   Deallocate local tiles (may be empty).                                           */
  /**********************************************************************************************************************/
  // AlignedAllocator takes care of this
}

void ParallelHeatSolver::initHaloExchange() {
  /**********************************************************************************************************************/
  /*                            Initialize variables and MPI datatypes for halo exchange.                               */
  /*                    If mSimulationProps.isRunParallelRMA() flag is set to true, create RMA windows.                 */
  /**********************************************************************************************************************/
  std::array<int, 2> halo_send_start_up = {haloZoneSize, haloZoneSize};
  std::array<int, 2> halo_send_start_down = {tile_size_y, haloZoneSize};
  std::array<int, 2> halo_send_start_left = {haloZoneSize, haloZoneSize};
  std::array<int, 2> halo_send_start_right = {haloZoneSize, tile_size_x};

  std::array<int, 2> halo_receive_start_up = {0, haloZoneSize};
  std::array<int, 2> halo_receive_start_down = {tile_size_y + (int)haloZoneSize, haloZoneSize};
  std::array<int, 2> halo_receive_start_left = {haloZoneSize, 0};
  std::array<int, 2> halo_receive_start_right = {haloZoneSize, tile_size_x + (int)haloZoneSize};

  std::array<int, 2> halo_dims_row = {haloZoneSize, tile_size_x};
  std::array<int, 2> halo_dims_col = {tile_size_y, haloZoneSize};

  // halo send row up int
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_send_start_up.data(), MPI_ORDER_C, MPI_INT, &halo_send_row_up_type_int);
  MPI_Type_commit(&halo_send_row_up_type_int);
  // halo send row down int
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_send_start_down.data(), MPI_ORDER_C, MPI_INT, &halo_send_row_down_type_int);
  MPI_Type_commit(&halo_send_row_down_type_int);
  // halo send row up float
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_send_start_up.data(), MPI_ORDER_C, MPI_FLOAT, &halo_send_row_up_type_float);
  MPI_Type_commit(&halo_send_row_up_type_float);
  // halo send row down float
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_send_start_down.data(), MPI_ORDER_C, MPI_FLOAT, &halo_send_row_down_type_float);
  MPI_Type_commit(&halo_send_row_down_type_float);
  // halo send col left int
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_send_start_left.data(), MPI_ORDER_C, MPI_INT, &halo_send_col_left_type_int);
  MPI_Type_commit(&halo_send_col_left_type_int);
  // halo send col right int
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_send_start_right.data(), MPI_ORDER_C, MPI_INT, &halo_send_col_right_type_int);
  MPI_Type_commit(&halo_send_col_right_type_int);
  // halo send col left float
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_send_start_left.data(), MPI_ORDER_C, MPI_FLOAT, &halo_send_col_left_type_float);
  MPI_Type_commit(&halo_send_col_left_type_float);
  // halo send col right float
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_send_start_right.data(), MPI_ORDER_C, MPI_FLOAT, &halo_send_col_right_type_float);
  MPI_Type_commit(&halo_send_col_right_type_float);

  ///////////////////////////////

  // halo receive row up int
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_receive_start_up.data(), MPI_ORDER_C, MPI_INT, &halo_receive_row_up_type_int);
  MPI_Type_commit(&halo_receive_row_up_type_int);
  // halo receive row down int
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_receive_start_down.data(), MPI_ORDER_C, MPI_INT, &halo_receive_row_down_type_int);
  MPI_Type_commit(&halo_receive_row_down_type_int);
  // halo receive row up float
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_receive_start_up.data(), MPI_ORDER_C, MPI_FLOAT, &halo_receive_row_up_type_float);
  MPI_Type_commit(&halo_receive_row_up_type_float);
  // halo receive row down float
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_row.data(), halo_receive_start_down.data(), MPI_ORDER_C, MPI_FLOAT, &halo_receive_row_down_type_float);
  MPI_Type_commit(&halo_receive_row_down_type_float);
  // halo receive col left int
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_receive_start_left.data(), MPI_ORDER_C, MPI_INT, &halo_receive_col_left_type_int);
  MPI_Type_commit(&halo_receive_col_left_type_int);
  // halo receive col right int
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_receive_start_right.data(), MPI_ORDER_C, MPI_INT, &halo_receive_col_right_type_int);
  MPI_Type_commit(&halo_receive_col_right_type_int);
  // halo receive col lereceive_ft float
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_receive_start_left.data(), MPI_ORDER_C, MPI_FLOAT, &halo_receive_col_left_type_float);
  MPI_Type_commit(&halo_receive_col_left_type_float);
  // halo receive col right float
  MPI_Type_create_subarray(2, local_tile_with_halo_dims.data(), halo_dims_col.data(), halo_receive_start_right.data(), MPI_ORDER_C, MPI_FLOAT, &halo_receive_col_right_type_float);
  MPI_Type_commit(&halo_receive_col_right_type_float);
  
  if (mSimulationProps.isRunParallelRMA()) {
    MPI_Win_create(tile_temps[NEW].data(), tile_size_with_halo_x * tile_size_with_halo_y * sizeof(float), sizeof(float), MPI_INFO_NULL, cart_comm, &wins[NEW]);
    MPI_Win_fence(0, wins[NEW]);
    
    MPI_Win_create(tile_temps[OLD].data(), tile_size_with_halo_x * tile_size_with_halo_y * sizeof(float), sizeof(float), MPI_INFO_NULL, cart_comm, &wins[OLD]);
    MPI_Win_fence(0, wins[OLD]);
  }
}

void ParallelHeatSolver::deinitHaloExchange() {
  /**********************************************************************************************************************/
  /*                            Deinitialize variables and MPI datatypes for halo exchange.                             */
  /**********************************************************************************************************************/
  MPI_Type_free(&halo_send_row_up_type_int);
  MPI_Type_free(&halo_send_row_down_type_int);
  MPI_Type_free(&halo_send_row_up_type_float);
  MPI_Type_free(&halo_send_row_down_type_float);

  MPI_Type_free(&halo_send_col_left_type_int);
  MPI_Type_free(&halo_send_col_right_type_int);
  MPI_Type_free(&halo_send_col_left_type_float);
  MPI_Type_free(&halo_send_col_right_type_float);

  MPI_Type_free(&halo_receive_row_up_type_int);
  MPI_Type_free(&halo_receive_row_down_type_int);
  MPI_Type_free(&halo_receive_row_up_type_float);
  MPI_Type_free(&halo_receive_row_down_type_float);

  MPI_Type_free(&halo_receive_col_left_type_int);
  MPI_Type_free(&halo_receive_col_right_type_int);
  MPI_Type_free(&halo_receive_col_left_type_float);
  MPI_Type_free(&halo_receive_col_right_type_float);
  
  if (mSimulationProps.isRunParallelRMA()) {
    MPI_Win_free(&wins[NEW]);
    MPI_Win_free(&wins[OLD]);
  }
}

template <typename T>
void ParallelHeatSolver::scatterTiles(const T *globalData, T *localData) {
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported scatter datatype!");

  /**********************************************************************************************************************/
  /*                      Implement master's global tile scatter to each rank's local tile.                             */
  /*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
  /**********************************************************************************************************************/
  const MPI_Datatype global_tile_type = std::is_same_v<T, int> ? global_tile_type_int : global_tile_type_float;
  const MPI_Datatype local_tile_type = std::is_same_v<T, int> ? local_tile_type_int : local_tile_type_float;

  MPI_Scatterv(globalData, counts.get(), displacements.get(), global_tile_type, localData, 1, local_tile_type, 0, MPI_COMM_WORLD);
}

template <typename T>
void ParallelHeatSolver::gatherTiles(const T *localData, T *globalData) {
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported gather datatype!");

  /**********************************************************************************************************************/
  /*                      Implement each rank's local tile gather to master's rank global tile.                         */
  /*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
  /**********************************************************************************************************************/
  const MPI_Datatype global_tile_type = std::is_same_v<T, int> ? global_tile_type_int : global_tile_type_float;
  const MPI_Datatype local_tile_type = std::is_same_v<T, int> ? local_tile_type_int : local_tile_type_float;

  MPI_Gatherv(localData, 1, local_tile_type, globalData, counts.get(), displacements.get(), global_tile_type, 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::computeHaloZones(const float *oldTemp, float *newTemp) {
  /**********************************************************************************************************************/
  /*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
  /*                        Use updateTile method to compute new temperatures in halo zones.                            */
  /*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
  /**********************************************************************************************************************/
  // in 1D decomp, the ranks at the ends do not calculate the sides at all (Dirichlet boundary condition of the simulation)
  if ((neighbors[ND::W] == MPI_PROC_NULL || neighbors[ND::E] == MPI_PROC_NULL) && tile_size_x <= int(haloZoneSize)) return;
  
  //       ________
  //     __|______|___
  //  __|__|_top__|__|__
  // |  |le|      |ri|  |
  // |  |ft|      |gh|  |
  // |__|__|______|t_|__|
  //    |__|bottom|__|
  //       |______|
  
  // centers of the sides
  if (!is_top) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize * 2, haloZoneSize, tile_size_x - (2 * haloZoneSize), haloZoneSize, tile_size_with_halo_x);
  }
  
  if (!is_bottom) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize * 2, tile_size_y, tile_size_x - (2 * haloZoneSize), haloZoneSize, tile_size_with_halo_x);
  }
  
  if (!is_left) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize, haloZoneSize * 2, haloZoneSize, tile_size_y - (2 * haloZoneSize), tile_size_with_halo_x);
  }
  
  if (!is_right) { 
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), tile_size_x, haloZoneSize * 2, haloZoneSize, tile_size_y - (2 * haloZoneSize), tile_size_with_halo_x);
  }

  // small square tiles in the corners
  if (!is_top && !is_left) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize, haloZoneSize, haloZoneSize, haloZoneSize, tile_size_with_halo_x);
  }
  
  if (!is_top && !is_right) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), tile_size_x, haloZoneSize, haloZoneSize, haloZoneSize, tile_size_with_halo_x);
  }
  
  if (!is_bottom && !is_left) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), haloZoneSize, tile_size_y, haloZoneSize, haloZoneSize, tile_size_with_halo_x);
  }

  if (!is_bottom && !is_right) {
    updateTile(oldTemp, newTemp, tile_params.data(), tile_map.data(), tile_size_x, tile_size_y, haloZoneSize, haloZoneSize, tile_size_with_halo_x);
  }
}

// local data should have size (tile_size_with_halo_x * tile_size_with_halo_y)
void ParallelHeatSolver::startHaloExchangeP2P(float *localData, std::array<MPI_Request, 8> &requests) {
  /**********************************************************************************************************************/
  /*                       Start the non-blocking halo zones exchange using P2P communication.                          */
  /*                         Use the requests array to return the requests from the function.                           */
  /*                            Don't forget to set the empty requests to MPI_REQUEST_NULL.                             */
  /**********************************************************************************************************************/
  // left col float, receive in right col float
  MPI_Isend(localData, 1, halo_send_col_left_type_float, neighbors[ND::W], TO_W, cart_comm, &requests[0]);
  MPI_Irecv(localData, 1, halo_receive_col_right_type_float, neighbors[ND::E], FROM_E, cart_comm, &requests[4]);
  
  // right col float, receive in left col float
  MPI_Isend(localData, 1, halo_send_col_right_type_float, neighbors[ND::E], TO_E, cart_comm, &requests[1]);
  MPI_Irecv(localData, 1, halo_receive_col_left_type_float, neighbors[ND::W], FROM_W, cart_comm, &requests[5]);

  // up row float, receive in down col float
  MPI_Isend(localData, 1, halo_send_row_up_type_float, neighbors[ND::N], TO_N, cart_comm, &requests[2]);
  MPI_Irecv(localData, 1, halo_receive_row_down_type_float, neighbors[ND::S], FROM_S, cart_comm, &requests[6]);

  // down row float, receive in up col float
  MPI_Isend(localData, 1, halo_send_row_down_type_float, neighbors[ND::S], TO_S, cart_comm, &requests[3]);
  MPI_Irecv(localData, 1, halo_receive_row_up_type_float, neighbors[ND::N], FROM_N, cart_comm, &requests[7]);
}

void ParallelHeatSolver::startHaloExchangeRMA(float *localData, MPI_Win window) {
  /**********************************************************************************************************************/
  /*                       Start the non-blocking halo zones exchange using RMA communication.                          */
  /*                   Do not forget that you put/get the values to/from the target's opposite side                     */
  /**********************************************************************************************************************/
  // open the window for exchange
  MPI_Win_fence(0, window);

  // send to west neighbor
  if (neighbors[ND::W] != MPI_PROC_NULL) {
    MPI_Put(localData, 1, halo_send_col_left_type_float, neighbors[ND::W], 0, 1, halo_receive_col_right_type_float, window);
  }
  
  // send to north neighbor
  if (neighbors[ND::N] != MPI_PROC_NULL) {
    MPI_Put(localData, 1, halo_send_row_up_type_float, neighbors[ND::N], 0, 1, halo_receive_row_down_type_float, window);
  }

  // send to east neighbor
  if (neighbors[ND::E] != MPI_PROC_NULL) {
    MPI_Put(localData, 1, halo_send_col_right_type_float, neighbors[ND::E], 0, 1, halo_receive_col_left_type_float, window);
  }

  // send to south neighbor
  if (neighbors[ND::S] != MPI_PROC_NULL) {
    MPI_Put(localData, 1, halo_send_row_down_type_float, neighbors[ND::S], 0, 1, halo_receive_row_up_type_float, window);
  }
}

void ParallelHeatSolver::awaitHaloExchangeP2P(std::array<MPI_Request, 8> &requests) {
  /**********************************************************************************************************************/
  /*                       Wait for all halo zone exchanges to finalize using P2P communication.                        */
  /**********************************************************************************************************************/
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void ParallelHeatSolver::awaitHaloExchangeRMA(MPI_Win window) {
  /**********************************************************************************************************************/
  /*                       Wait for all halo zone exchanges to finalize using RMA communication.                        */
  /**********************************************************************************************************************/
  MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPRECEDE, window);
}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>> &outResult) {
  /**********************************************************************************************************************/
  /*                                         Scatter initial data.                                                      */
  /**********************************************************************************************************************/
  std::array<MPI_Request, 8> requestsP2P{};
  
  // broadcasting the material and simulation properties
  heater_temp = mMaterialProps.getHeaterTemperature();
  cooler_temp = mMaterialProps.getCoolerTemperature();
  MPI_Bcast(&heater_temp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cooler_temp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  interation_count = mSimulationProps.getNumIterations();
  is_p2p_mode = mSimulationProps.isRunParallelP2P() ? 1 : 0;
  MPI_Bcast(&interation_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&is_p2p_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  // scatter the data from the input file
  scatterTiles(mMaterialProps.getDomainParameters().data(), tile_params.data());
  scatterTiles(mMaterialProps.getDomainMap().data(), tile_map.data());
  scatterTiles(mMaterialProps.getInitialTemperature().data(), tile_temps[OLD].data());

  /**********************************************************************************************************************/
  /* Exchange halo zones of initial domain temperature and parameters using P2P communication. Wait for them to finish. */
  /**********************************************************************************************************************/
  // exchange termperature
  startHaloExchangeP2P(tile_temps[OLD].data(), requestsP2P);
  awaitHaloExchangeP2P(requestsP2P);

  // exchange parameters
  startHaloExchangeP2P(tile_params.data(), requestsP2P);
  awaitHaloExchangeP2P(requestsP2P);

  /**********************************************************************************************************************/
  /*                            Copy initial temperature to the second buffer.                                          */
  /**********************************************************************************************************************/
  tile_temps[NEW] = tile_temps[OLD];

  double startTime = MPI_Wtime();

  // 3. Start main iterative simulation loop.
  for (std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter) {
    const std::size_t oldIdx = iter % 2;       // Index of the buffer with old temperatures
    const std::size_t newIdx = (iter + 1) % 2; // Index of the buffer with new temperatures

    /**********************************************************************************************************************/
    /*                            Compute and exchange halo zones using P2P or RMA.                                       */
    /**********************************************************************************************************************/
    computeHaloZones(tile_temps[oldIdx].data(), tile_temps[newIdx].data());

    if (mSimulationProps.isRunParallelP2P()) {
      startHaloExchangeP2P(tile_temps[newIdx].data(), requestsP2P);
    } else {
      startHaloExchangeRMA(tile_temps[newIdx].data(), wins[newIdx]);
    }
    
    /**********************************************************************************************************************/
    /*                           Compute the rest of the tile. Use updateTile method.                                     */
    /**********************************************************************************************************************/
    updateTile(tile_temps[oldIdx].data(), tile_temps[newIdx].data(), tile_params.data(), tile_map.data(), 2 * haloZoneSize, 2 * haloZoneSize,
      (tile_size_x - 2 * haloZoneSize), tile_size_y - (2 * haloZoneSize), 2 * haloZoneSize + tile_size_x);

    /**********************************************************************************************************************/
    /*                            Wait for all halo zone exchanges to finalize.                                           */
    /**********************************************************************************************************************/
    if (mSimulationProps.isRunParallelP2P()) {
      awaitHaloExchangeP2P(requestsP2P);
    } else {
      awaitHaloExchangeRMA(wins[newIdx]);
    }

    if (shouldStoreData(iter)) {
      /**********************************************************************************************************************/
      /*                          Store the data into the output file using parallel or sequential IO.                      */
      /**********************************************************************************************************************/
      if (mSimulationProps.useParallelIO()) {
        storeDataIntoFileParallel(mFileHandle, iter, tile_temps[NEW].data());
      } else {
        gatherTiles(tile_temps[newIdx].data(), outResult.data());
        if (mWorldRank == 0) {
          // only world comm root rank has the correct gathered temperatures
          storeDataIntoFileSequential(mFileHandle, iter, outResult.data());
        }
      }
    }

    if (shouldPrintProgress(iter) && should_compute_average) {
      /**********************************************************************************************************************/
      /*                 Compute and print middle column average temperature and print progress report.                     */
      /**********************************************************************************************************************/
      float par_avg = computeMiddleColumnAverageTemperatureParallel(tile_temps[newIdx].data());

      // only rank in center col comm has the correct reduced value to report
      if (mCenterColCommRank == 0) {
        printProgressReport(iter, par_avg);
      }
    }
  }

  const std::size_t resIdx = mSimulationProps.getNumIterations() % 2; // Index of the buffer with final temperatures
  double elapsedTime = MPI_Wtime() - startTime;

  /**********************************************************************************************************************/
  /*                                     Gather final domain temperature.                                               */
  /**********************************************************************************************************************/
  gatherTiles(tile_temps[resIdx].data(), outResult.data());

  /**********************************************************************************************************************/
  /*           Compute (sequentially) and report final middle column temperature average and print final report.        */
  /**********************************************************************************************************************/
  if (mWorldRank == 0) {
    float seq_avg = computeMiddleColumnAverageTemperatureSequential(outResult.data());
    printFinalReport(elapsedTime, seq_avg);
  }
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const {
  /**********************************************************************************************************************/
  /*                Return true if rank should compute middle column average temperature.                               */
  /**********************************************************************************************************************/

  if (decomp == 1) return (dims[0] / 2 == cart_coords[0]);
  return (dims[1] / 2 == cart_coords[1]);
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureParallel(const float *localData) const {
  /**********************************************************************************************************************/
  /*                  Implement parallel middle column average temperature computation.                                 */
  /*                      Use OpenMP directives to accelerate the local computations.                                   */
  /**********************************************************************************************************************/
  int col_index = ((global_edge_size / 2) % tile_size_x) + haloZoneSize;
  
  float sum = 0.0f, total_sum{0.0f};
  
  #pragma omp simd reduction(+:sum) aligned(localData: 64) simdlen(16)
  for (int i = haloZoneSize; i < (int)(tile_size_y + haloZoneSize); i++) {
    sum += localData[i * tile_size_with_halo_x + col_index];
  }

  MPI_Reduce(&sum, &total_sum, 1, MPI_FLOAT, MPI_SUM, 0, center_col_comm);
  return total_sum / global_edge_size;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureSequential(const float *globalData) const {
  /**********************************************************************************************************************/
  /*                  Implement sequential middle column average temperature computation.                               */
  /*                      Use OpenMP directives to accelerate the local computations.                                   */
  /**********************************************************************************************************************/
  float sum{.0f};
  int col_index = global_edge_size / 2;
  #pragma omp simd reduction(+:sum) aligned(globalData: 64) simdlen(16)
  for (int i = 0; i < global_edge_size; i++) {
    sum += globalData[i * global_edge_size + col_index];
  }

  return sum / global_edge_size;
}

void ParallelHeatSolver::openOutputFileSequential() {
  mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                          H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (!mFileHandle.valid()) {
    throw std::ios::failure("Cannot create output file!");
  }
}

void ParallelHeatSolver::storeDataIntoFileSequential(hid_t fileHandle,
                                                     std::size_t iteration,
                                                     const float *globalData) {
  storeDataIntoFile(fileHandle, iteration, globalData);
}

void ParallelHeatSolver::openOutputFileParallel() {
#ifdef H5_HAVE_PARALLEL

  /**********************************************************************************************************************/
  /*                          Open output HDF5 file for parallel access with alignment.                                 */
  /*      Set up faplHandle to use MPI-IO and alignment. The handle will automatically release the resource.            */
  /**********************************************************************************************************************/
  Hdf5PropertyListHandle faplHandle(H5Pcreate(H5P_FILE_ACCESS));
  herr_t status_mpio = H5Pset_fapl_mpio(faplHandle, cart_comm, MPI_INFO_NULL);

  if(status_mpio < 0) {
    fprintf(stderr, "Parallel IO: Failed to set parallel IO for cartesian topology\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
    
  herr_t status_align = H5Pset_alignment(faplHandle, 0, 512*1024);

  if(status_align < 0) {
    fprintf(stderr, "Parallel IO: Failed to set alignment\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                          H5F_ACC_TRUNC,
                          H5P_DEFAULT,
                          faplHandle);
  
  if (!mFileHandle.valid()) {
    throw std::ios::failure("Cannot create output file!");
  }

  // optimisation to only calculate it once,
  // it is used in every parallel write
  // compute the tile offsets and sizes
  if (decomp == 1) {
    tileOffset[0] = 0;
    tileOffset[1] = cart_coords[0] * tile_size_x;
  } else {
    tileOffset[0] = cart_coords[0] * tile_size_y;
    tileOffset[1] = cart_coords[1] * tile_size_x;
  }
  // tile sizes for parallel I/O
  local_tile_size_with_halo[0] = tile_size_with_halo_y;
  local_tile_size_with_halo[1] = tile_size_with_halo_x;
  // sizes in an hsize_t array
  local_tile_size[0] = tile_size_y;
  local_tile_size[1] = tile_size_x;
  // offset in an hsize_t array
  start_tile_halo[0] = haloZoneSize;
  start_tile_halo[1] = haloZoneSize;
#else
  throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}

void ParallelHeatSolver::storeDataIntoFileParallel(hid_t fileHandle,
                                                   [[maybe_unused]] std::size_t iteration,
                                                   [[maybe_unused]] const float *localData) {
  if (fileHandle == H5I_INVALID_HID) {
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
    Hdf5PropertyListHandle datasetPropListHandle(H5Pcreate(H5P_DATASET_CREATE));

    /**********************************************************************************************************************/
    /*                            Create dataset property list to set up chunking.                                        */
    /*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
    /**********************************************************************************************************************/
    H5Pset_chunk(datasetPropListHandle, 2, local_tile_size.data());

    Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
    Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                              H5T_NATIVE_FLOAT, dataSpaceHandle,
                                              H5P_DEFAULT, datasetPropListHandle,
                                              H5P_DEFAULT));
    
    /**********************************************************************************************************************/
    /*                Create memory dataspace representing tile in the memory (set up memSpaceHandle).                    */
    /**********************************************************************************************************************/
    Hdf5DataspaceHandle memSpaceHandle(H5Screate_simple(2, local_tile_size_with_halo.data(), nullptr));

    /**********************************************************************************************************************/
    /*              Select inner part of the tile in memory and matching part of the dataset in the file                  */
    /*                           (given by position of the tile in global domain).                                        */
    /**********************************************************************************************************************/
    H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, start_tile_halo.data(), nullptr, local_tile_size.data(), nullptr);
    H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, tileOffset.data(), nullptr, local_tile_size.data(), nullptr);

    /**********************************************************************************************************************/
    /*              Perform collective write operation, writting tiles from all processes at once.                        */
    /*                                   Set up the propListHandle variable.                                              */
    /**********************************************************************************************************************/
    Hdf5PropertyListHandle propListHandle(H5Pcreate(H5P_DATASET_XFER));
    H5Pset_dxpl_mpio(propListHandle, H5FD_MPIO_COLLECTIVE);

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