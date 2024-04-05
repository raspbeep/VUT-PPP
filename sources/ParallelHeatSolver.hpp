/**
 * @file    ParallelHeatSolver.hpp
 * 
 * @author  Name Surname <xlogin00@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#ifndef PARALLEL_HEAT_SOLVER_HPP
#define PARALLEL_HEAT_SOLVER_HPP

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include <mpi.h>

#include "AlignedAllocator.hpp"
#include "Hdf5Handle.hpp"
#include "HeatSolverBase.hpp"

// Neighbor directions for indexing in neighbor array
enum ND {
  W  = 0,
  N  = 1,
  E  = 2,
  S  = 3
};

constexpr int OLD = 0;
constexpr int NEW = 1;

constexpr int TO_N = 0;
constexpr int TO_S = 1;
constexpr int TO_E = 2;
constexpr int TO_W = 3;

constexpr int FROM_N = 1;
constexpr int FROM_S = 0;
constexpr int FROM_E = 3;
constexpr int FROM_W = 2;

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 2D block grid decomposition.
 */
class ParallelHeatSolver : public HeatSolverBase
{
  public:
    /**
     * @brief Constructor - Initializes the solver. This includes:
     *        - Construct 2D grid of tiles.
     *        - Create MPI datatypes used in the simulation.
     *        - Open SEQUENTIAL or PARALLEL HDF5 file.
     *        - Allocate data for local tiles.
     *
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    ParallelHeatSolver(const SimulationProperties& simulationProps, const MaterialProperties& materialProps);
    
    /// @brief Inherit constructors from the base class.
    using HeatSolverBase::HeatSolverBase;

    /**
     * @brief Destructor - Releases all resources allocated by the solver.
     */
    virtual ~ParallelHeatSolver() override;

    /// @brief Inherit assignment operator from the base class.
    using HeatSolverBase::operator=;

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: The vector is allocated (and should be used) *ONLY*
     *                        by master process (rank 0 in MPI_COMM_WORLD)!
     */
    virtual void run(std::vector<float, AlignedAllocator<float>>& outResult) override;

  protected:
    int n_dims;
    std::array<int, 2> dims;
    std::array<int, 2> local_tile_with_halo_dims;
    int total_size;
    MPI_Comm cart_comm;
    int cart_rank;
    std::array<int, 2> cart_coords;
    std::array<int, 4> neighbors;

    MPI_Comm center_col_comm;

    std::unique_ptr<int[]> counts;
    std::unique_ptr<int[]> displacements;
    float heater_temp;
    float cooler_temp;
    int interation_count;
    int is_p2p_mode;

    int global_edge_size;
    int tile_size_x, tile_size_y;
    int tile_size_with_halo_x, tile_size_with_halo_y;

    MPI_Datatype global_tile_type_float;
    MPI_Datatype global_tile_type_int;
    
    // local tile type (with halo borders)
    MPI_Datatype local_tile_type_int;
    // local tile type (with halo borders)
    MPI_Datatype local_tile_type_float;
    
    // hallo exchange types
    MPI_Datatype halo_send_row_up_type_int;
    MPI_Datatype halo_send_row_up_type_float;

    MPI_Datatype halo_send_row_down_type_int;
    MPI_Datatype halo_send_row_down_type_float;
    
    MPI_Datatype halo_send_col_left_type_int;
    MPI_Datatype halo_send_col_left_type_float;

    MPI_Datatype halo_send_col_right_type_int;
    MPI_Datatype halo_send_col_right_type_float;
  
    MPI_Datatype halo_receive_row_up_type_int;
    MPI_Datatype halo_receive_row_up_type_float;

    MPI_Datatype halo_receive_row_down_type_int;
    MPI_Datatype halo_receive_row_down_type_float;
    
    MPI_Datatype halo_receive_col_left_type_int;
    MPI_Datatype halo_receive_col_left_type_float;

    MPI_Datatype halo_receive_col_right_type_int;
    MPI_Datatype halo_receive_col_right_type_float;

    AlignedAllocator<int> intAllocator;
    AlignedAllocator<float> floatAllocator;

    std::vector<float, AlignedAllocator<float>> domain;

    std::vector<int, AlignedAllocator<float>> tile_map;
    std::vector<float, AlignedAllocator<float>> tile_params;
    std::array<std::vector<float, AlignedAllocator<float>>, 2> tile_temps;

  private:
    // TODO: DELETE ME
    void print_global_tile_float(const float *arr);
    void print_global_tile_int(const int *arr);
    void print_local_tile_no_halo_float(const float *arr);
    void print_local_tile_no_halo_int(const int *arr);
    void print_local_tile_float(const float *arr);
    void print_local_tile_int(const int *arr);

    /**
     * @brief Get type of the code.
     * @return Returns type of the code.
     */
    std::string_view getCodeType() const override;

    /**
     * @brief Initialize the grid topology.
     */
    void initGridTopology();

    /**
     * @brief Deinitialize the grid topology.
     */
    void deinitGridTopology();

    /**
     * @brief Initialize variables and MPI datatypes for data scattering and gathering.
     */
    void initDataDistribution();

    /**
     * @brief Deinitialize variables and MPI datatypes for data scattering and gathering.
     */
    void deinitDataDistribution();

    /**
     * @brief Allocate memory for local tiles.
     */
    void allocLocalTiles();

    /**
     * @brief Deallocate memory for local tiles.
     */
    void deallocLocalTiles();

    /**
     * @brief Initialize variables and MPI datatypes for halo exchange.
     */
    void initHaloExchange();

    /**
     * @brief Deinitialize variables and MPI datatypes for halo exchange.
     */
    void deinitHaloExchange();

    /**
     * @brief Scatter global data to local tiles.
     * @tparam T Type of the data to be scattered. Must be either float or int.
     * @param globalData Global data to be scattered.
     * @param localData  Local data to be filled with scattered values.
     */
    template<typename T>
    void scatterTiles(const T* globalData, T* localData);

    /**
     * @brief Gather local tiles to global data.
     * @tparam T Type of the data to be gathered. Must be either float or int.
     * @param localData  Local data to be gathered.
     * @param globalData Global data to be filled with gathered values.
     */
    template<typename T>
    void gatherTiles(const T* localData, T* globalData);

    /**
     * @brief Compute temperature of the next iteration in the halo zones.
     * @param oldTemp Old temperature values.
     * @param newTemp New temperature values.
     */
    void computeHaloZones(const float* oldTemp, float* newTemp);

    /**
     * @brief Start halo exchange using point-to-point communication.
     * @param localData Local data to be exchanged.
     * @param request   Array of MPI_Request objects to be filled with requests.
     */
    void startHaloExchangeP2P(float* localData, std::array<MPI_Request, 8>& request);

    /**
     * @brief Await halo exchange using point-to-point communication.
     * @param request Array of MPI_Request objects to be awaited.
     */
    void awaitHaloExchangeP2P(std::array<MPI_Request, 8>& request);

    /**
     * @brief Start halo exchange using RMA communication.
     * @param localData Local data to be exchanged.
     * @param window    MPI_Win object to be used for RMA communication.
     */
    void startHaloExchangeRMA(float* localData, MPI_Win window);

    /**
     * @brief Await halo exchange using RMA communication.
     * @param window MPI_Win object to be used for RMA communication.
     */
    void awaitHaloExchangeRMA(MPI_Win window);

    /**
     * @brief Computes global average temperature of middle column across
     *        processes in "mGridMiddleColComm" communicator.
     *        NOTE: All ranks in the communicator *HAVE* to call this method.
     * @param localData Data of the local tile.
     * @return Returns average temperature over middle of all tiles in the communicator.
     */
    float computeMiddleColumnAverageTemperatureParallel(const float* localData) const;

    /**
     * @brief Computes global average temperature of middle column of the domain
     *        using values collected to MASTER rank.
     *        NOTE: Only single RANK needs to call this method.
     * @param globalData Simulation state collected to the MASTER rank.
     * @return Returns the average temperature.
     */
    float computeMiddleColumnAverageTemperatureSequential(const float* globalData) const;

    /**
     * @brief Opens output HDF5 file for sequential access by MASTER rank only.
     *        NOTE: Only MASTER (rank = 0) should call this method.
     */
    void openOutputFileSequential();

    /**
     * @brief Stores current state of the simulation into the output file.
     *        NOTE: Only MASTER (rank = 0) should call this method.
     * @param fileHandle HDF5 file handle to be used for the writting operation.
     * @param iteration  Integer denoting current iteration number.
     * @param data       Square 2D array of edgeSize x edgeSize elements containing
     *                   simulation state to be stored in the file.
     */
    void storeDataIntoFileSequential(hid_t fileHandle, std::size_t iteration, const float* globalData);

    /**
     * @brief Opens output HDF5 file for parallel/cooperative access.
     *        NOTE: This method *HAS* to be called from all processes in the communicator.
     */
    void openOutputFileParallel();

    /**
     * @brief Stores current state of the simulation into the output file.
     *        NOTE: All processes which opened the file HAVE to call this method collectively.
     * @param fileHandle HDF5 file handle to be used for the writting operation.
     * @param iteration  Integer denoting current iteration number.
     * @param localData  Local 2D array (tile) of mLocalTileSize[0] x mLocalTileSize[1] elements
     *                   to be stored at tile specific position in the output file.
     *                   This method skips halo zones of the tile and stores only relevant data.
     */
    void storeDataIntoFileParallel(hid_t fileHandle, std::size_t iteration, const float* localData);

    /**
     * @brief Determines if the process should compute average temperature of the middle column.
     * @return Returns true if the process should compute average temperature of the middle column.
     */
    bool shouldComputeMiddleColumnAverageTemperature() const;

    /// @brief Code type string.
    static constexpr std::string_view codeType{"par"};

    /// @brief Size of the halo zone.
    static constexpr std::size_t haloZoneSize{2};

    /// @brief Process rank in the global communicator (MPI_COMM_WORLD).
    int mWorldRank{};

    /// @brief Total number of processes in MPI_COMM_WORLD.
    int mWorldSize{};

    /// @brief Output file handle (parallel or sequential).
    Hdf5FileHandle mFileHandle{};
};

#endif /* PARALLEL_HEAT_SOLVER_HPP */
