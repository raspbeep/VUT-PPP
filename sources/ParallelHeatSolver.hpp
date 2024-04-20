/**
 * @file    ParallelHeatSolver.hpp
 * 
 * @author  Pavel Kratochvíl <xkrato61@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-04-26
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
  private:
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

    /// @brief rank the center col communicator
    int mCenterColCommRank{};

    /// @brief rank the cartesian topology communicator
    int mCartRank{};

    /// @brief Output file handle (parallel or sequential).
    Hdf5FileHandle mFileHandle{};
    
    /// @brief count of processors in x direction
    int nx;

    /// @brief count of processors in y direction
    int ny;
    
    /// @brief number of dimensions for decomposition
    int n_dims;
    
    /// @brief decomposition type
    int decomp;
    
    /// @brief true if the rank contains the middle column
    bool should_compute_average;
    
    /// @brief dimensions of the topology (in 1d, only the first value used)
    std::array<int, 2> dims;
    
    /// @brief for storage cartesion coordinates
    std::array<int, 2> cart_coords;
    
    /// @brief storage of neighbor ranks in all directions
    std::array<int, 4> neighbors;
    
    /// @brief offset of the local data in the local array (which contains halo zones around it)
    std::array<int, 2> start_arr;
    
    /// @brief dimensions of the local tile without halo zones
    std::array<int, 2> local_tile_dims;
    
    /// @brief dimensions of the local tile with halo border (x + 2) * (y + 2)
    std::array<int, 2> local_tile_with_halo_dims;

    /// @brief cartesian topology communicator
    MPI_Comm cart_comm;
    
    /// @brief center column communication for middle column average calculation
    MPI_Comm center_col_comm;
    
    /// @brief indications of the location for computeHaloZones()
    bool is_top, is_bottom, is_left, is_right;

    /// @brief array containing counts of tiles for each rannk (full of ones)
    std::unique_ptr<int[]> counts;
    
    /// @brief array containing displacements of individual tiles in the global domain
    std::unique_ptr<int[]> displacements;
    
    /// @brief for broadcasting initial parameters
    float heater_temp;
    
    /// @brief for broadcasting initial parameters
    float cooler_temp;
    
    /// @brief for broadcasting initial parameters
    int interation_count;
    
    /// @brief for broadcasting initial parameters
    int is_p2p_mode;

    /// @brief size of the entire domain
    int global_edge_size;
    
    /// @brief local tile size
    int tile_size_x, tile_size_y;
    
    /// @brief tile size with 
    int tile_size_with_halo_x, tile_size_with_halo_y;
    
    /// @brief tile for distribution from root rank
    MPI_Datatype global_tile_type_float;
    
    /// @brief tile for distribution from root rank
    MPI_Datatype global_tile_type_int;
    
    /// @brief local int tile type (with halo borders)
    MPI_Datatype local_tile_type_int;
    
    /// @brief local tile float type (with halo borders)
    MPI_Datatype local_tile_type_float;
    
    /// @brief halo exchange type
    MPI_Datatype halo_send_row_up_type_int;
    
    /// @brief halo exchange type
    MPI_Datatype halo_send_row_up_type_float;

    /// @brief halo exchange type
    MPI_Datatype halo_send_row_down_type_int;
    
    /// @brief halo exchange type
    MPI_Datatype halo_send_row_down_type_float;
    
    /// @brief halo exchange type
    MPI_Datatype halo_send_col_left_type_int;
    
    /// @brief halo exchange type
    MPI_Datatype halo_send_col_left_type_float;

    /// @brief halo exchange type
    MPI_Datatype halo_send_col_right_type_int;
    
    /// @brief halo exchange type
    MPI_Datatype halo_send_col_right_type_float;
  
    /// @brief halo exchange type
    MPI_Datatype halo_receive_row_up_type_int;
    
    /// @brief halo exchange type
    MPI_Datatype halo_receive_row_up_type_float;

    /// @brief halo exchange type
    MPI_Datatype halo_receive_row_down_type_int;
    
    /// @brief halo exchange type
    MPI_Datatype halo_receive_row_down_type_float;
    
    /// @brief halo exchange type
    MPI_Datatype halo_receive_col_left_type_int;
    
    /// @brief halo exchange type
    MPI_Datatype halo_receive_col_left_type_float;

    /// @brief halo exchange type
    MPI_Datatype halo_receive_col_right_type_int;
    
    /// @brief halo exchange type
    MPI_Datatype halo_receive_col_right_type_float;

    /// @brief windows for RMA data exchange
    std::array<MPI_Win, 2> wins;
    
    /// @brief aligned memory for array of tile map
    std::vector<int, AlignedAllocator<float>> tile_map;
    
    /// @brief aligned memory for array of tile params
    std::vector<float, AlignedAllocator<float>> tile_params;
    
    /// @brief aligned memory for 2d array of local tile temperatures
    std::array<std::vector<float, AlignedAllocator<float>>, 2> tile_temps;
    
    /// @brief tile size for dataset
    std::array<hsize_t, 2> local_tile_size;
    
    /// @brief tile size for memory space
    std::array<hsize_t, 2> local_tile_size_with_halo;
    
    /// @brief offset in data space hyperslab
    std::array<hsize_t, 2> tileOffset;
    
    /// @brief offset in memory space hyperslab
    std::array<hsize_t, 2> start_tile_halo;
};

#endif /* PARALLEL_HEAT_SOLVER_HPP */
